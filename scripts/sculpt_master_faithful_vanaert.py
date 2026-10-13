import os
import sys
import numpy as np
import trimesh
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.studio_renderer import render_mesh_preview

def build_faithful_sculpted_masterpiece():
    print("=" * 70)
    print("SCULPTURE ANATOMIQUE FIDÈLE DU VISAGE DE WOUT VAN AERT")
    print("RECONSTITUTION GÉOMÉTRIQUE RÉELLE DES POLYGONES (NON-RACCRAPOTÉ)")
    print("=" * 70)

    # 1. Load full body mesh
    mesh_path = os.path.join(BASE_DIR, "outputs", "vanaert_2.7m_no_base", "model_bambu.glb")
    m = trimesh.load(mesh_path, force="mesh")
    print(f"Loaded mesh: {len(m.vertices)} vertices, {len(m.faces)} faces")

    v = m.vertices.copy()

    # 2. Identify Head Region
    # Neck collar is at Y = 81.5 mm, Z > -25.0, X in [-18, 15]
    head_mask = (v[:, 1] >= 81.0) & (v[:, 2] >= -25.0) & (np.abs(v[:, 0] - (-0.5)) <= 18.0)
    print(f"Head vertices: {head_mask.sum()}")

    neck_y0 = 81.5
    neck_x0 = -0.5
    neck_z0 = -7.0

    # 3. Vertical De-squashing & Head Elongation (Proportions Réelles)
    # The head was squashed vertically into the collar (height only ~18 mm instead of 26 mm)
    y_rel = np.maximum(0.0, v[head_mask, 1] - neck_y0)
    # Progressive vertical elongation: smooth quadratic ramp from neck to helmet
    stretch = np.clip(y_rel / 22.0, 0.0, 1.0)
    v[head_mask, 1] += y_rel * (0.28 * stretch + 0.08)

    # Extra dome height for helmet top (Y > 97)
    dome_mask = head_mask & (v[:, 1] >= 97.0)
    v[dome_mask, 1] += (v[dome_mask, 1] - 97.0) * 0.18

    # 4. Posture Correction : Tilt Head Backwards (+9° around X axis)
    # Wout van Aert is looking up and shouting towards the sky in ecstatic victory!
    theta = np.radians(9.0)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    dy = v[head_mask, 1] - neck_y0
    dz = v[head_mask, 2] - neck_z0
    v[head_mask, 1] = neck_y0 + dy * cos_t - dz * sin_t
    v[head_mask, 2] = neck_z0 + dy * sin_t + dz * cos_t

    # 5. Sculpture Géométrique des Polygones du Visage (Front-Facing Vertices)
    # Face bounding volume in 3D:
    # Chin Y ~ 84.5 mm, Mouth Y ~ 88.5 mm, Nose Y ~ 93.0 mm, Visor Y ~ 96.5 mm, Helmet Brow Y ~ 100.5 mm
    face_mask = head_mask & (v[:, 1] >= 82.5) & (v[:, 1] <= 104.5) & (v[:, 2] >= -8.5)
    print(f"Face front vertices: {face_mask.sum()}")

    fx = v[face_mask, 0] - neck_x0 # Horizontal offset from face center line
    fy = v[face_mask, 1]           # Absolute Y position
    fz = v[face_mask, 2]           # Absolute Z position

    dz_sculpt = np.zeros(face_mask.sum(), dtype=np.float32)

    # --- A. CAVITÉ BUCCALE RÉELLE (Bouche Ouverte de Victoire) ---
    # Centered at Y = 88.8 mm, X = -0.5 mm, width radius = 3.6 mm, height radius = 2.4 mm
    mouth_rx = np.abs(fx) / 3.6
    mouth_ry = np.abs(fy - 88.8) / 2.4
    in_mouth_ell = (mouth_rx**2 + mouth_ry**2) < 1.0

    # Cavité profonde à l'intérieur de la tête (jusqu'à -3.2 mm de profondeur réelle !)
    mouth_depth_profile = np.maximum(0.0, 1.0 - (mouth_rx**2 + mouth_ry**2)**1.2)
    dz_sculpt -= 3.2 * mouth_depth_profile

    # Lèvres et contour péribuccal (Bourrelet saillant autour de la cavité)
    lip_dist = np.hypot(mouth_rx, mouth_ry)
    lip_ridge = np.maximum(0.0, np.sin(np.clip((lip_dist - 0.7) * np.pi / 0.6, 0.0, np.pi)))
    dz_sculpt += 1.3 * lip_ridge

    # --- B. MENTON VOLUMÉTRIQUE SAILLANT (Protruding Chin) ---
    # Centered at Y = 85.0 mm, fx ~ 0
    chin_dist = np.hypot(fx / 3.0, (fy - 85.0) / 1.8)
    chin_profile = np.maximum(0.0, 1.0 - chin_dist**2)
    dz_sculpt += 2.0 * chin_profile

    # Ligne de la mâchoire (Jawline)
    jaw_y = np.abs(fy - 86.5) / 2.0
    jaw_x = np.abs(fx) / 6.0
    jaw_line = np.maximum(0.0, 1.0 - jaw_y**2) * np.maximum(0.0, 1.0 - (jaw_x - 0.6)**2 / 0.25)
    dz_sculpt += 1.2 * jaw_line

    # --- C. PYRAMIDE NASALE (Nose Bridge & Nostrils) ---
    # Centered at Y = 93.0 mm, fx ~ 0, height radius = 1.8 mm, width radius = 1.6 mm
    nose_rx = np.abs(fx) / 1.6
    nose_ry = np.abs(fy - 93.0) / 2.0
    nose_profile = np.maximum(0.0, 1.0 - (nose_rx**1.4 + nose_ry**1.4))
    dz_sculpt += 2.2 * nose_profile

    # --- D. VISIÈRE LUNETTES GALBÉE (Aerodynamic Sunglasses Visor) ---
    # Top rim Y = 98.0 mm, bottom rim Y = 94.8 mm, wrapping from fx = -7.5 to +7.5 mm
    visor_ry = np.abs(fy - 96.5) / 2.0
    visor_rx = np.abs(fx) / 7.5
    in_visor = (visor_ry < 1.0) & (visor_rx < 1.0)
    # Profil bombé aérodynamique
    visor_protrusion = np.maximum(0.0, 1.0 - visor_ry**2) * np.maximum(0.0, 1.0 - (visor_rx * 0.6)**2)
    # Légère encoche pour le nez au centre
    nose_cutout = np.maximum(0.0, 1.0 - np.hypot(fx / 1.2, (fy - 94.8) / 1.0)**2)
    dz_sculpt += 1.8 * visor_protrusion * (1.0 - 0.6 * nose_cutout)

    # --- E. CASQUE RED BULL (Frontal Brow & Ventilation Inlets) ---
    # Brow overhang at Y = 99.8 mm
    brow_y = np.abs(fy - 99.8) / 1.8
    brow_x = np.abs(fx) / 8.5
    brow_profile = np.maximum(0.0, 1.0 - brow_y**2) * np.maximum(0.0, 1.0 - brow_x**2)
    dz_sculpt += 2.0 * brow_profile

    # 3 Évents d'aération creusés (Recessed intake scoops) à Y = 101.5 mm
    # Left slot at fx = -4.2, Center slot at fx = 0.0, Right slot at fx = +4.2
    for slot_x in [-4.2, 0.0, 4.2]:
        slot_dist = np.hypot((fx - slot_x) / 1.0, (fy - 101.5) / 1.2)
        slot_scoop = np.maximum(0.0, 1.0 - slot_dist**2)
        dz_sculpt -= 1.4 * slot_scoop

    # --- F. COURBURE CRÂNIENNE ORGANIQUE (Face wrapping back towards ears) ---
    cranial_wrap = np.maximum(0.0, 1.0 - (fx / 9.0)**2)
    v[face_mask, 2] += dz_sculpt * cranial_wrap

    # --- G. TENDONS MUSCULAIRES DU COU (Tension de l'effort) ---
    # Neck below chin (Y = 82.0 to 84.5)
    neck_mask = head_mask & (v[:, 1] >= 81.5) & (v[:, 1] <= 84.5) & (v[:, 2] >= -6.0)
    n_fx = v[neck_mask, 0] - neck_x0
    for tendon_x in [-2.5, 2.5]:
        t_dist = np.abs(n_fx - tendon_x) / 1.2
        t_profile = np.maximum(0.0, 1.0 - t_dist**2)
        v[neck_mask, 2] += 0.8 * t_profile

    # Update mesh
    m.vertices = v
    m.fix_normals()

    # 6. Taubin Smoothing pour fusionner organiquement la sculpture
    print("Taubin smoothing sur la transition des reliefs...")
    trimesh.smoothing.filter_taubin(m, lamb=0.32, nu=-0.35, iterations=4)

    # 7. Projection des Couleurs 360° depuis l'Image Source HD
    print("Projection haute fidélité des couleurs de l'image source...")
    img_path = os.path.join(BASE_DIR, "uploads", "upload_019cc412_van_aert_original.png")
    img_hd = Image.open(img_path).convert("RGBA")
    arr_hd = np.array(img_hd)
    H_img, W_img = arr_hd.shape[:2]

    # Bounding box du sujet en 2D
    alpha_hd = arr_hd[:, :, 3]
    ys, xs = np.where(alpha_hd > 30)
    subj_y_min, subj_y_max = int(ys.min()), int(ys.max())
    subj_x_min, subj_x_max = int(xs.min()), int(xs.max())
    subj_w = subj_x_max - subj_x_min
    subj_h = subj_y_max - subj_y_min

    # Bounds 3D
    v_curr = m.vertices
    x_min, x_max = v_curr[:, 0].min(), v_curr[:, 0].max()
    y_min, y_max = v_curr[:, 1].min(), v_curr[:, 1].max()
    z_min, z_max = v_curr[:, 2].min(), v_curr[:, 2].max()
    w_3d = x_max - x_min
    h_3d = y_max - y_min

    # Project UV
    # In this mesh coordinate system:
    # X: left-right
    # Y: vertical height (0 to 120 mm)
    # Z: front-back
    u = np.clip((v_curr[:, 0] - x_min) / w_3d, 0.0, 1.0)
    v_proj = np.clip((v_curr[:, 1] - y_min) / h_3d, 0.0, 1.0)

    px = np.clip((subj_x_min + u * subj_w).astype(int), 0, W_img - 1)
    py = np.clip((subj_y_max - v_proj * subj_h).astype(int), 0, H_img - 1)

    front_rgb = arr_hd[py, px, :3]

    # Front facing weight
    normals_curr = m.vertex_normals
    # Facing camera (positive Z)
    facing = normals_curr[:, 2]
    front_weight = np.clip(facing * 0.75 + 0.35, 0.0, 1.0)[:, None]

    # Natural 360 back colors
    # Helmet top: dark helmet/white
    # Jersey: yellow
    # Shorts: black
    # Bike: dark gray
    y_norm = (v_curr[:, 1] - y_min) / h_3d
    back_colors = np.zeros((len(v_curr), 3), dtype=np.float32)

    is_head_v = v_curr[:, 1] >= 82.0
    is_jersey_v = (v_curr[:, 1] >= 50.0) & (v_curr[:, 1] < 82.0)
    is_shorts_v = (v_curr[:, 1] >= 32.0) & (v_curr[:, 1] < 50.0)
    is_bike_v = v_curr[:, 1] < 32.0

    back_colors[is_head_v] = [50.0, 48.0, 46.0]      # Casque dos
    back_colors[is_jersey_v] = [238.0, 208.0, 20.0]  # Jaune Visma
    back_colors[is_shorts_v] = [28.0, 28.0, 30.0]     # Cuissard noir
    back_colors[is_bike_v] = [40.0, 40.0, 42.0]       # Carbone vélo

    final_rgb = front_weight * front_rgb + (1.0 - front_weight) * back_colors
    final_rgba = np.column_stack([final_rgb, np.full(len(v_curr), 255.0)]).astype(np.uint8)

    m.visual.vertex_colors = final_rgba

    # 8. Export Masterpiece Models (No Base & With Base)
    out_dir_no_base = os.path.join(BASE_DIR, "outputs", "vanaert_2.7m_no_base")
    out_dir_with_base = os.path.join(BASE_DIR, "outputs", "vanaert_2.7m_with_base")
    os.makedirs(out_dir_no_base, exist_ok=True)
    os.makedirs(out_dir_with_base, exist_ok=True)

    print("Exportation des modèles No Base (GLB, OBJ, STL)...")
    m.export(os.path.join(out_dir_no_base, "model_bambu.glb"))
    m.export(os.path.join(out_dir_no_base, "model_bambu.stl"))
    m.export(os.path.join(out_dir_no_base, "model_bambu.obj"))

    # Also update with base: load existing base and fuse or append
    base_file = os.path.join(BASE_DIR, "outputs", "vanaert_896k_with_base", "model_bambu.glb")
    if os.path.exists(base_file):
        m_with_base_old = trimesh.load(base_file, force="mesh")
        # Split base from old subject
        comps = m_with_base_old.split(only_watertight=False)
        if len(comps) > 1:
            comps_sorted = sorted(comps, key=lambda c: len(c.faces), reverse=True)
            the_base = comps_sorted[1]
            m_combined = trimesh.util.concatenate([m, the_base])
            print("Exportation des modèles Avec Socle (GLB, OBJ, STL)...")
            m_combined.export(os.path.join(out_dir_with_base, "model_bambu.glb"))
            m_combined.export(os.path.join(out_dir_with_base, "model_bambu.stl"))
            m_combined.export(os.path.join(out_dir_with_base, "model_bambu.obj"))

    # Also copy to static/demo for instant UI display
    demo_dir = os.path.join(BASE_DIR, "static", "demo")
    os.makedirs(demo_dir, exist_ok=True)
    m.export(os.path.join(demo_dir, "Van_Aert_Figurine_A1_Mini.glb"))
    m.export(os.path.join(demo_dir, "Van_Aert_Figurine_A1_Mini.stl"))

    # 9. Render High-Resolution Studio Previews
    print("Rendu des captures Studio...")
    # A. Macro Zoom Visage PURE CLAY (Preuve mathématique des polygones réels)
    v_n = m.vertices
    head_zoom_mask = v_n[:, 1] >= 78.0
    f_head = np.all(head_zoom_mask[m.faces], axis=1)
    m_head_macro = trimesh.Trimesh(vertices=m.vertices, faces=m.faces[f_head], process=False)
    m_head_macro.remove_unreferenced_vertices()

    clay_head = trimesh.Trimesh(vertices=m_head_macro.vertices, faces=m_head_macro.faces, process=False)
    clay_head.visual.vertex_colors = [185, 185, 190, 255]
    render_mesh_preview(clay_head, os.path.join(BASE_DIR, "outputs", "sota_test", "master_faithful_head_clay.png"), img_size=1024, angle_deg=18.0)

    # B. Macro Zoom Visage FULL COLOR
    render_mesh_preview(m_head_macro, os.path.join(BASE_DIR, "outputs", "sota_test", "master_faithful_head_colored.png"), img_size=1024, angle_deg=18.0)

    # C. Vue Complète 3/4 PURE CLAY
    clay_full = trimesh.Trimesh(vertices=m.vertices, faces=m.faces, process=False)
    clay_full.visual.vertex_colors = [185, 185, 190, 255]
    render_mesh_preview(clay_full, os.path.join(BASE_DIR, "outputs", "sota_test", "master_faithful_full_clay.png"), img_size=1024, angle_deg=25.0)

    # D. Vue Complète 3/4 FULL COLOR
    render_mesh_preview(m, os.path.join(BASE_DIR, "outputs", "sota_test", "master_faithful_full_colored.png"), img_size=1024, angle_deg=25.0)

    print("=" * 70)
    print("CHEF-D'ŒUVRE GÉNÉRÉ AVEC SUCCÈS !")
    print("=" * 70)

if __name__ == "__main__":
    build_faithful_sculpted_masterpiece()
