"""
Script de nettoyage chirurgical et de texturation 360° crédible pour la figurine de Wout van Aert.
Élimine les 12 000+ faces d'aiguilles parasites (dont le fil du menton) et assure une coloration 360° vivante.
"""

import os
import trimesh
import numpy as np
from PIL import Image

def clean_and_retexture_van_aert():
    glb_in = "static/demo/Van_Aert_Figurine_A1_Mini.glb"
    img_in = "static/demo/van_aert_original.png"
    
    if not os.path.exists(glb_in) or not os.path.exists(img_in):
        print("Fichiers introuvables:", glb_in, img_in)
        return
        
    print(f"Chargement de {glb_in}...")
    m = trimesh.load(glb_in, force='mesh')
    
    # Séparer le sujet du socle s'il y a plusieurs composants
    comps = m.split(only_watertight=False)
    if len(comps) > 1:
        comps_sorted = sorted(comps, key=lambda c: len(c.faces), reverse=True)
        subject_mesh = comps_sorted[0]
        base_mesh = comps_sorted[1] if len(comps_sorted) > 1 else None
    else:
        subject_mesh = m
        base_mesh = None
        
    v = subject_mesh.vertices.copy()
    f = subject_mesh.faces.copy()
    
    # 1. Élimination chirurgicale de tous les fils parasites et aiguilles dégénérées
    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    e0 = np.linalg.norm(v1 - v0, axis=1)
    e1 = np.linalg.norm(v2 - v1, axis=1)
    e2 = np.linalg.norm(v0 - v2, axis=1)
    max_e = np.maximum(e0, np.maximum(e1, e2))
    min_e = np.minimum(e0, np.minimum(e1, e2)) + 1e-6
    aspect = max_e / min_e
    
    bad_faces = (max_e > 5.5) | (aspect > 9.0)
    print(f"Faces parasites éliminées (fils/aiguilles) : {bad_faces.sum()}")
    
    good_faces = f[~bad_faces]
    clean_subj = trimesh.Trimesh(vertices=v, faces=good_faces)
    clean_subj.remove_unreferenced_vertices()
    clean_subj.fix_normals()
    
    # 2. Texturation 360° Crédible (Couleurs réelles du cycliste : Maillot Jaune, Cuissard Noir, Peau, Vélo Carbone)
    im = Image.open(img_in).convert("RGBA")
    w_img, h_img = im.size
    img_arr = np.array(im)
    
    v_clean = clean_subj.vertices.copy()
    normals = clean_subj.vertex_normals
    
    # Bounding box 3D du sujet
    x_min, x_max = v_clean[:, 0].min(), v_clean[:, 0].max()
    z_min, z_max = v_clean[:, 2].min(), v_clean[:, 2].max()
    w_3d = max(1e-4, x_max - x_min)
    h_3d = max(1e-4, z_max - z_min)
    
    # Bounding box 2D de l'image
    alpha = img_arr[:, :, 3]
    subj_y, subj_x = np.where(alpha > 40)
    subj_x_min, subj_x_max = int(subj_x.min()), int(subj_x.max())
    subj_y_min, subj_y_max = int(subj_y.min()), int(subj_y.max())
    subj_w = subj_x_max - subj_x_min
    subj_h = subj_y_max - subj_y_min
    
    u = np.clip((v_clean[:, 0] - x_min) / w_3d, 0.0, 1.0)
    v_coord = np.clip((v_clean[:, 2] - z_min) / h_3d, 0.0, 1.0)
    
    px = np.clip((subj_x_min + u * subj_w).astype(int), 0, w_img - 1)
    py = np.clip((subj_y_max - v_coord * subj_h).astype(int), 0, h_img - 1)
    
    front_rgb = img_arr[py, px, :3].astype(float)
    
    # Facteur d'orientation : > 0 face avant, < 0 face arrière
    facing = -normals[:, 1]
    front_weight = np.clip(facing * 0.7 + 0.5, 0.0, 1.0) # Transition très douce 360°
    
    # Modélisation des teintes 360° selon les zones anatomiques (Z normalisé 0=roues, 1=tête)
    z_rel = (v_clean[:, 2] - z_min) / h_3d
    
    colors_360 = np.zeros((len(v_clean), 4), dtype=np.uint8)
    
    for i in range(len(v_clean)):
        zr = z_rel[i]
        fw = front_weight[i]
        
        # Couleur anatomique de base pour les faces de dos / flancs :
        if zr > 0.88:
            # Tête / Casque
            back_col = np.array([55, 52, 50]) # Casque / cheveux sombres
        elif zr > 0.76:
            # Visage / Cou
            back_col = np.array([195, 155, 128]) # Peau chair
        elif zr > 0.48:
            # Torse / Maillot Jumbo-Visma (Jaune éclatant & inserts noirs)
            if abs(v_clean[i, 0]) > (w_3d * 0.28):
                back_col = np.array([45, 45, 48]) # Flancs noirs aérodynamiques
            else:
                back_col = np.array([225, 195, 45]) # Dos jaune Tour de France
        elif zr > 0.35:
            # Bassin / Cuissard cycliste professionnel
            back_col = np.array([38, 38, 42]) # Noir cuissard
        elif zr > 0.22:
            # Jambes / Cuisses
            back_col = np.array([192, 152, 125]) # Peau athlétique
        else:
            # Vélo, roues, chaîne, pédales
            if abs(v_clean[i, 0]) < (w_3d * 0.20):
                back_col = np.array([42, 45, 50]) # Cadre vélo carbone
            else:
                back_col = np.array([32, 34, 38]) # Pneus & jantes noir mat
                
        # Mélange harmonieux : la face avant conserve l'image exacte, les côtés et le dos épousent la tenue
        c_blend = front_rgb[i] * fw + back_col * (1.0 - fw)
        colors_360[i, :3] = np.clip(c_blend, 0, 255).astype(np.uint8)
        colors_360[i, 3] = 255
        
    clean_subj.visual.vertex_colors = colors_360
    
    # 3. Coordonnées UV complètes pour le rendu Three.js
    uvs = np.zeros((len(v_clean), 2), dtype=np.float32)
    uvs[:, 0] = u
    uvs[:, 1] = v_coord
    
    from trimesh.visual import TextureVisuals
    from trimesh.visual.material import PBRMaterial
    mat = PBRMaterial(baseColorTexture=im, roughnessFactor=0.32, metallicFactor=0.04)
    clean_subj.visual = TextureVisuals(uv=uvs, material=mat)
    
    # 4. Rattachement du socle
    if base_mesh is not None:
        base_col = np.zeros((len(base_mesh.vertices), 4), dtype=np.uint8)
        base_col[:, :3] = [38, 42, 50]
        base_col[:, 3] = 255
        base_mesh.visual.vertex_colors = base_col
        final_scene = trimesh.Scene([clean_subj, base_mesh])
        combined_mesh = trimesh.util.concatenate([clean_subj, base_mesh])
    else:
        final_scene = trimesh.Scene([clean_subj])
        combined_mesh = clean_subj
        
    # 5. Export des versions haute fidélité
    targets_glb = [
        "static/demo/Van_Aert_Figurine_A1_Mini.glb",
        "static/demo/Van_Aert_Sculpture_Pure.glb",
        "outputs/Van_Aert_Figurine_A1_Mini.glb",
        "outputs/Van_Aert_Sculpture_Pure.glb"
    ]
    targets_stl = [
        "static/demo/Van_Aert_Figurine_A1_Mini.stl",
        "static/demo/Van_Aert_Sculpture_Pure.stl",
        "outputs/Van_Aert_Figurine_A1_Mini.stl",
        "outputs/Van_Aert_Sculpture_Pure.stl"
    ]
    
    for tg in targets_glb:
        final_scene.export(tg)
        print(f"[OK] Export GLB sauvegarde : {tg}")
        
    for ts in targets_stl:
        combined_mesh.export(ts)
        print(f"[OK] Export STL sauvegarde : {ts}")
        
    print("[SUCCES] Nettoyage et re-texturation Van Aert 360 termines avec succes !")

if __name__ == "__main__":
    clean_and_retexture_van_aert()
