"""
Rendu Studio Haute Définition pour maillages 3D.
Génère une vue de prévisualisation studio éclairée (Lambert / Phong) en nuances de gris ou bronze.
"""

import numpy as np
from PIL import Image
import trimesh

def render_mesh_preview(
    mesh: trimesh.Trimesh,
    output_png_path: str,
    img_size: int = 512,
    angle_deg: float = 25.0
) -> str:
    """
    Rend une projection 3D orthographique éclairée du maillage avec ombrage diffus (Lambert).
    """
    # Copie du maillage pour transformation
    m = mesh.copy()
    
    # Recentrage
    m.apply_translation(-m.centroid)
    
    # Rotation pour vue 3/4 élégante
    if angle_deg != 0:
        rot_matrix = trimesh.transformations.rotation_matrix(
            np.radians(angle_deg), [0, 1, 0]
        )
        m.apply_transform(rot_matrix)
        
    # Légère inclinaison vers le haut
    tilt_matrix = trimesh.transformations.rotation_matrix(
        np.radians(-10), [1, 0, 0]
    )
    m.apply_transform(tilt_matrix)

    # Récupération sommets et faces
    vertices = m.vertices
    faces = m.faces
    normals = m.face_normals
    
    # Boîte englobante
    min_xy = np.min(vertices[:, :2], axis=0)
    max_xy = np.max(vertices[:, :2], axis=0)
    span = np.max(max_xy - min_xy)
    if span < 1e-4:
        span = 1.0
        
    # Normalisation dans l'espace écran (0 à img_size - 1)
    scale = (img_size * 0.82) / span
    proj_verts = (vertices[:, :2] - (min_xy + max_xy) / 2.0) * scale + (img_size / 2.0)
    depths = vertices[:, 2]
    
    # Éclairage directionnel (Lumière studio venant d'en haut à gauche)
    light_dir = np.array([0.4, 0.6, 0.7])
    light_dir = light_dir / np.linalg.norm(light_dir)
    
    # Tri des triangles par profondeur Z pour rendu Z-buffer
    face_centers_z = np.mean(depths[faces], axis=1)
    sorted_face_indices = np.argsort(face_centers_z)
    
    # Image buffer et Z buffer
    buffer = np.ones((img_size, img_size, 3), dtype=np.uint8) * 32  # Fond studio sombre (#202020)
    z_buffer = np.full((img_size, img_size), -np.inf, dtype=np.float32)
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        
        fig, ax = plt.subplots(figsize=(6, 6), dpi=img_size // 6)
        fig.patch.set_facecolor('#1e1e24')
        ax.set_facecolor('#1e1e24')
        
        # Calcul des couleurs par face avec éclairage studio diffus (Lambertian shading)
        dot_products = np.clip(np.dot(normals, light_dir), 0.25, 1.0)
        
        if hasattr(m.visual, 'vertex_colors') and m.visual.vertex_colors is not None:
            v_col = m.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
            face_col = np.mean(v_col[faces], axis=1)
            colors = np.clip(face_col * (0.35 + 0.75 * dot_products[:, None]), 0.0, 1.0)
            colors = np.column_stack([colors, np.ones(len(faces), dtype=np.float32)])
        else:
            # Teinte Bambu PLA Gris Studio
            colors = np.zeros((len(faces), 4), dtype=np.float32)
            colors[:, 0] = np.clip(dot_products * 0.80 + 0.1, 0.0, 1.0)  # R
            colors[:, 1] = np.clip(dot_products * 0.83 + 0.1, 0.0, 1.0)  # G
            colors[:, 2] = np.clip(dot_products * 0.87 + 0.1, 0.0, 1.0)  # B
            colors[:, 3] = 1.0
        
        # Extraction des polygones 2D
        polys = proj_verts[faces]
        
        collection = PolyCollection(
            polys[sorted_face_indices],
            facecolors=colors[sorted_face_indices],
            edgecolors='none',
            antialiased=True
        )
        
        ax.add_collection(collection)
        ax.set_xlim(0, img_size)
        ax.set_ylim(0, img_size)
        ax.axis('off')
        
        plt.tight_layout(pad=0)
        plt.savefig(output_png_path, dpi=img_size // 6, facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    except Exception as e:
        print(f"[StudioRenderer] Avertissement matplotlib ({e}), fallback image de prévisualisation neutre")
        img = Image.new('RGB', (img_size, img_size), color=(30, 30, 36))
        img.save(output_png_path)
    
    return output_png_path
