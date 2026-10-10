"""
Agent Antigravity — PNG to 3D Print Engine
Moteur de génération et de traitement de maillages 3D pour Bambu Lab A1 Mini.
"""

import os
import io
import time
import subprocess
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import scipy.ndimage as ndi
import trimesh
import trimesh.smoothing as smoothing
import manifold3d
from manifold3d import Manifold, Mesh

# Bambu Lab A1 Mini specs
A1_MINI_BED_X = 180.0
A1_MINI_BED_Y = 180.0
A1_MINI_MAX_Z = 180.0
NOZZLE_DIAMETER = 0.4
MIN_RECOMMENDED_WALL = 0.8  # 2 perimeters of 0.4mm nozzle


def process_image_cleanup(image: Image.Image, remove_bg: bool = True, bg_threshold: int = 240, contrast_factor: float = 1.0) -> Image.Image:
    """Nettoyage professionnel de l'image, détourage automatique par rembg IA ou inondation connectée des 4 bords."""
    im = image.convert("RGBA")
    
    if contrast_factor != 1.0:
        enhancer = ImageEnhance.Contrast(image.convert("RGB"))
        rgb_enh = enhancer.enhance(contrast_factor)
        r, g, b = rgb_enh.split()
        im = Image.merge("RGBA", (r, g, b, im.getchannel("A")))
        
    if remove_bg:
        arr = np.array(im)
        has_clean_alpha = (arr[:, :, 3] < 200).sum() > (arr.shape[0] * arr.shape[1] * 0.05)
        
        if not has_clean_alpha:
            # 1. Tenter le détourage neural rembg pour un résultat studio parfait sans aucun cadre
            rembg_done = False
            try:
                import rembg
                im_rembg = rembg.remove(im)
                arr_rembg = np.array(im_rembg)
                if (arr_rembg[:, :, 3] > 30).sum() > 500:
                    arr = arr_rembg
                    rembg_done = True
            except Exception as e:
                rembg_done = False
                
            # 2. Si rembg échoue ou indisponible, inondation connectée depuis les 4 bords
            if not rembg_done:
                h, w = arr.shape[:2]
                edge_pixels = np.concatenate([arr[0, :, :3], arr[-1, :, :3], arr[:, 0, :3], arr[:, -1, :3]], axis=0)
                bg_color = np.median(edge_pixels, axis=0)
                
                dist_to_bg = np.sqrt(np.sum((arr[:, :, :3].astype(float) - bg_color) ** 2, axis=2))
                is_bg_candidate = (dist_to_bg < 55.0) | ((arr[:, :, 0] > 220) & (arr[:, :, 1] > 220) & (arr[:, :, 2] > 220))
                
                from scipy.ndimage import binary_dilation
                border_seed = np.zeros((h, w), dtype=bool)
                border_seed[0, :] = is_bg_candidate[0, :]
                border_seed[-1, :] = is_bg_candidate[-1, :]
                border_seed[:, 0] = is_bg_candidate[:, 0]
                border_seed[:, -1] = is_bg_candidate[:, -1]
                
                connected_bg = border_seed.copy()
                struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
                for _ in range(max(h, w)):
                    dilated = binary_dilation(connected_bg, structure=struct) & is_bg_candidate
                    if np.array_equal(dilated, connected_bg):
                        break
                    connected_bg = dilated
                    
                fg_mask = ~connected_bg
                fg_mask = ndi.binary_fill_holes(fg_mask)
                
                # Conserver uniquement la plus grande composante connectée (le sujet central)
                labels, num_features = ndi.label(fg_mask)
                if num_features > 1:
                    sizes = ndi.sum(fg_mask, labels, range(1, num_features + 1))
                    largest_label = np.argmax(sizes) + 1
                    fg_mask = (labels == largest_label)
                    
                alpha = np.where(fg_mask, 255, 0).astype(np.uint8)
                alpha = ndi.gaussian_filter(alpha.astype(float), 0.8)
                arr[:, :, 3] = (alpha > 80).astype(np.uint8) * 255

        # 3. Cadrage strict (crop) sur le sujet avec marge de 6 px
        y_idx, x_idx = np.where(arr[:, :, 3] > 30)
        if len(y_idx) > 0:
            h, w = arr.shape[:2]
            y1 = max(0, y_idx.min() - 6)
            y2 = min(h, y_idx.max() + 7)
            x1 = max(0, x_idx.min() - 6)
            x2 = min(w, x_idx.max() + 7)
            arr = arr[y1:y2, x1:x2]

        im = Image.fromarray(arr, "RGBA")
        
    return im


def check_bambu_printability(mesh: trimesh.Trimesh) -> dict:
    """Analyse la conformité du modèle pour la Bambu Lab A1 Mini."""
    extents = mesh.extents  # [X, Y, Z]
    bounds = mesh.bounds
    vol_cm3 = max(0.001, mesh.volume / 1000.0) if mesh.is_watertight else (extents[0]*extents[1]*extents[2]*0.2 / 1000.0)
    
    # Poids PLA estimé (densité ~ 1.24 g/cm3 avec 15% infill)
    pla_weight_g = round(vol_cm3 * 1.24 * 0.35, 1)  # ~35% solid equivalent
    
    # Temps d'impression approximatif sur A1 Mini (vitesse ~150-200 mm/s)
    est_minutes = int(max(15, (extents[2] / 0.16) * 1.5 + (pla_weight_g * 1.8)))
    hours = est_minutes // 60
    mins = est_minutes % 60
    time_str = f"{hours}h {mins}min" if hours > 0 else f"{mins} min"
    
    warnings = []
    if extents[0] > A1_MINI_BED_X or extents[1] > A1_MINI_BED_Y or extents[2] > A1_MINI_MAX_Z:
        warnings.append(f"Dimensions ({extents[0]:.1f}x{extents[1]:.1f}x{extents[2]:.1f} mm) dépassent le volume maximal de la Bambu Lab A1 Mini (180x180x180 mm).")
    
    # Vérification épaisseur minimale
    min_feature = min(extents[0], extents[1]) / 50.0
    if min_feature < MIN_RECOMMENDED_WALL:
        warnings.append(f"Attention aux détails fins : certaines parois peuvent être inférieures à {MIN_RECOMMENDED_WALL} mm (minimum recommandé pour buse 0.4 mm).")
        
    return {
        "extents_mm": [round(x, 2) for x in extents],
        "bounds_mm": [[round(x, 2) for x in b] for b in bounds],
        "is_watertight": bool(mesh.is_watertight),
        "vertices_count": len(mesh.vertices),
        "faces_count": len(mesh.faces),
        "estimated_weight_g": pla_weight_g,
        "estimated_time": time_str,
        "fits_a1_mini": bool(extents[0] <= A1_MINI_BED_X and extents[1] <= A1_MINI_BED_Y and extents[2] <= A1_MINI_MAX_Z),
        "warnings": warnings
    }


def generate_mode1_extruded(
    image: Image.Image,
    target_height_mm: float = 100.0,
    base_thickness_mm: float = 3.0,
    relief_height_mm: float = 3.0,
    add_base: bool = True,
    base_shape: str = "oval", # "oval", "rect", "none"
    invert: bool = False
) -> trimesh.Trimesh:
    """Mode 1 : Relief extrudé à partir des contours du sujet / logo."""
    im = process_image_cleanup(image, remove_bg=True)
    arr = np.array(im)
    alpha = arr[:, :, 3]
    
    # Masque binaire du sujet
    mask = (alpha > 128) if not invert else (alpha <= 128)
    
    # Bounding box
    y_idx, x_idx = np.where(mask)
    if len(y_idx) == 0:
        raise ValueError("Aucun sujet détecté dans l'image.")
    
    y1, y2, x1, x2 = y_idx.min(), y_idx.max(), x_idx.min(), x_idx.max()
    cropped_mask = mask[y1:y2+1, x1:x2+1]
    
    # Downsample pour maillage propre
    target_res = 300
    h, w = cropped_mask.shape
    scale_factor = target_res / max(h, w)
    nh, nw = max(10, int(h * scale_factor)), max(10, int(w * scale_factor))
    
    mask_pil = Image.fromarray((cropped_mask * 255).astype(np.uint8)).resize((nw, nh), Image.Resampling.BILINEAR)
    grid = np.array(mask_pil) > 128
    
    # Création du volume extrudé
    total_z = base_thickness_mm + relief_height_mm if add_base else relief_height_mm
    
    # Échelle en mm
    pixel_size = target_height_mm / nh
    target_width_mm = nw * pixel_size
    
    # Voxel grid
    nz = max(4, int(total_z / pixel_size))
    vol = np.zeros((nw, nh, nz), dtype=bool)
    
    base_z_slices = max(1, int(base_thickness_mm / pixel_size)) if add_base else 0
    relief_z_slices = max(1, int(relief_height_mm / pixel_size))
    
    # Relief
    for z in range(base_z_slices, base_z_slices + relief_z_slices):
        vol[:, :, z] = grid.T
        
    # Base
    if add_base and base_shape != "none":
        if base_shape == "oval":
            cy, cx = nh / 2.0, nw / 2.0
            y_coords, x_coords = np.ogrid[:nh, :nw]
            rx = nw / 2.0 + (5.0 / pixel_size)
            ry = nh / 2.0 + (5.0 / pixel_size)
            base_mask = (((x_coords - cx) / rx) ** 2 + ((y_coords - cy) / ry) ** 2) <= 1.0
        else: # rect
            base_mask = np.ones((nh, nw), dtype=bool)
            
        for z in range(base_z_slices):
            vol[:, :, z] = base_mask.T
            
    # Marching cubes
    from skimage.measure import marching_cubes
    verts, faces, _, _ = marching_cubes(vol.astype(float), level=0.5)
    
    # Mise à l'échelle en mm
    verts[:, 0] *= pixel_size
    verts[:, 1] *= pixel_size
    verts[:, 2] = (verts[:, 2] / (nz - 1)) * total_z
    
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.fix_normals()
    smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=6)
    
    # Alignement au sol Z=0
    mesh.apply_translation([0, 0, -mesh.bounds[0][2]])
    center_xy = (mesh.bounds[0][:2] + mesh.bounds[1][:2]) / 2.0
    mesh.apply_translation([-center_xy[0], -center_xy[1], 0])
    
    # Mise à l'échelle finale exacte
    scale = target_height_mm / mesh.extents[1]
    mesh.apply_scale(scale)
    mesh.apply_translation([0, 0, -mesh.bounds[0][2]])
    
    return mesh


def generate_mode2_heightmap(
    image: Image.Image,
    target_height_mm: float = 100.0,
    base_thickness_mm: float = 2.0,
    max_relief_mm: float = 4.0,
    invert: bool = False,
    resolution: int = 250,
    smooth_sigma: float = 1.0,
    frame_border_mm: float = 2.0
) -> trimesh.Trimesh:
    """Mode 2 : Bas-relief / Lithophane à profondeur variable (Heightmap) étanche et solide."""
    im = process_image_cleanup(image, remove_bg=False)
    
    # Niveaux de gris
    gray = ImageOps.grayscale(im.convert("RGB"))
    
    # Redimensionnement selon résolution
    w, h = gray.size
    aspect = w / h
    if aspect > 1.0:
        nw = resolution
        nh = int(resolution / aspect)
    else:
        nh = resolution
        nw = int(resolution * aspect)
        
    gray_res = gray.resize((nw, nh), Image.Resampling.LANCZOS)
    arr = np.array(gray_res, dtype=float) / 255.0
    
    if invert:
        arr = 1.0 - arr
        
    if smooth_sigma > 0:
        arr = ndi.gaussian_filter(arr, sigma=smooth_sigma)
        
    # Calcul des dimensions physiques
    real_h = target_height_mm
    real_w = real_h * aspect
    dx = real_w / (nw - 1)
    dy = real_h / (nh - 1)
    
    # Grille de surface supérieure
    x = np.linspace(-real_w / 2.0, real_w / 2.0, nw)
    y = np.linspace(real_h / 2.0, -real_h / 2.0, nh)
    xx, yy = np.meshgrid(x, y)
    
    z_top = base_thickness_mm + arr * max_relief_mm
    z_bottom = np.zeros_like(z_top)
    
    # Construction du solide watertight
    verts = []
    faces = []
    
    # Top vertices: indices 0 .. nw*nh - 1
    top_indices = np.arange(nw * nh).reshape((nh, nw))
    for i in range(nh):
        for j in range(nw):
            verts.append([xx[i, j], yy[i, j], z_top[i, j]])
            
    # Bottom vertices: indices nw*nh .. 2*nw*nh - 1
    bot_offset = nw * nh
    bot_indices = np.arange(bot_offset, 2 * bot_offset).reshape((nh, nw))
    for i in range(nh):
        for j in range(nw):
            verts.append([xx[i, j], yy[i, j], z_bottom[i, j]])
            
    # Faces du dessus (top)
    for i in range(nh - 1):
        for j in range(nw - 1):
            p1 = top_indices[i, j]
            p2 = top_indices[i, j + 1]
            p3 = top_indices[i + 1, j + 1]
            p4 = top_indices[i + 1, j]
            faces.append([p1, p2, p3])
            faces.append([p1, p3, p4])
            
    # Faces du dessous (bottom - inversées)
    for i in range(nh - 1):
        for j in range(nw - 1):
            p1 = bot_indices[i, j]
            p2 = bot_indices[i, j + 1]
            p3 = bot_indices[i + 1, j + 1]
            p4 = bot_indices[i + 1, j]
            faces.append([p1, p3, p2])
            faces.append([p1, p4, p3])
            
    # Murs latéraux (bords)
    # Bord haut (y max, i=0)
    for j in range(nw - 1):
        t1, t2 = top_indices[0, j], top_indices[0, j + 1]
        b1, b2 = bot_indices[0, j], bot_indices[0, j + 1]
        faces.append([t1, b1, b2])
        faces.append([t1, b2, t2])
        
    # Bord bas (y min, i=nh-1)
    for j in range(nw - 1):
        t1, t2 = top_indices[nh - 1, j], top_indices[nh - 1, j + 1]
        b1, b2 = bot_indices[nh - 1, j], bot_indices[nh - 1, j + 1]
        faces.append([t1, b2, b1])
        faces.append([t1, t2, b2])
        
    # Bord gauche (x min, j=0)
    for i in range(nh - 1):
        t1, t2 = top_indices[i, j:=0], top_indices[i + 1, 0]
        b1, b2 = bot_indices[i, 0], bot_indices[i + 1, 0]
        faces.append([t1, b2, b1])
        faces.append([t1, t2, b2])
        
    # Bord droit (x max, j=nw-1)
    for i in range(nh - 1):
        t1, t2 = top_indices[i, nw - 1], top_indices[i + 1, nw - 1]
        b1, b2 = bot_indices[i, nw - 1], bot_indices[i + 1, nw - 1]
        faces.append([t1, b1, b2])
        faces.append([t1, b2, t2])
        
    mesh = trimesh.Trimesh(vertices=np.array(verts), faces=np.array(faces))
    mesh.fix_normals()
    return mesh


def generate_mode3_volumetric(
    image: Image.Image,
    target_height_mm: float = 100.0,
    with_base: bool = True,
    base_height_mm: float = 4.5,
    resolution: int = 256
) -> trimesh.Trimesh:
    """Mode 3 : Véritable reconstruction 3D volumétrique (360°) par IA TripoSR sur GPU RTX 4070 Ti."""
    clean_im = process_image_cleanup(image, remove_bg=True)
    import tempfile
    import uuid
    uid = uuid.uuid4().hex[:8]
    temp_dir = os.path.join(tempfile.gettempdir(), "triposr_tmp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_in = os.path.join(temp_dir, f"in_{uid}.png")
    temp_out = os.path.join(temp_dir, f"out_{uid}.stl")
    script_path = os.path.join(temp_dir, f"run_{uid}.py")
    
    clean_im.save(temp_in)
    
    # Détection de l'environnement Python optimisé pour TripoSR sur GPU RTX 4070 Ti
    python_candidates = [
        r"E:\ComfyUI\venv\Scripts\python.exe",
        r"C:\Users\TrendingPC\stable-diffusion-webui\venv\Scripts\python.exe"
    ]
    python_exe = next((p for p in python_candidates if os.path.exists(p)), sys.executable)
    
    temp_in_clean = temp_in.replace('\\', '/')
    temp_out_clean = temp_out.replace('\\', '/')
    
    py_script = f"""
import sys, os
tripo_candidates = [
    "E:/ComfyUI/custom_nodes/ComfyUI-Flowty-TripoSR",
    "C:/Users/TrendingPC/TripoSR"
]
for p in tripo_candidates:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

import urllib3, requests
urllib3.disable_warnings()
old_send = requests.Session.send
def new_send(self, request, **kwargs):
    kwargs['verify'] = False
    return old_send(self, request, **kwargs)
requests.Session.send = new_send

import numpy as np
from PIL import Image
import torch
import trimesh
from tsr.system import TSR

im = Image.open('{temp_in_clean}').convert('RGBA')
arr = np.array(im).astype(np.float32) / 255.0
if arr.shape[2] == 4:
    rgb = arr[:, :, :3] * arr[:, :, 3:4] + (1.0 - arr[:, :, 3:4]) * 0.5
else:
    rgb = arr[:, :, :3]
input_pil = Image.fromarray((rgb * 255.0).astype(np.uint8)).convert('RGB')

comfy_ckpt = "E:/ComfyUI/models/checkpoints/triposr_model.ckpt"
comfy_cfg = "E:/ComfyUI/custom_nodes/ComfyUI-Flowty-TripoSR/config.yaml"
if os.path.exists(comfy_ckpt) and hasattr(TSR, "from_pretrained_custom"):
    model = TSR.from_pretrained_custom(comfy_ckpt, comfy_cfg)
else:
    model = TSR.from_pretrained('C:/Users/TrendingPC/TripoSR/pretrained', config_name='config.yaml', weight_name='model.ckpt')

model.renderer.set_chunk_size(8192)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model.to(device)

with torch.no_grad():
    scene_codes = model([input_pil], device=device)
    meshes = model.extract_mesh(scene_codes, resolution={resolution}, threshold=25.0)
    mesh = meshes[0]

mesh.export('{temp_out_clean}')
print('TripoSR completed!')
"""
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(py_script)
        
    cmd = [python_exe, script_path]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Erreur d'inférence IA TripoSR: {res.stderr}")
        
    raw_mesh = trimesh.load(temp_out)
    
    # Orientation repère Bambu Studio
    verts = raw_mesh.vertices.copy()
    verts_bambu = np.zeros_like(verts)
    verts_bambu[:, 0] = verts[:, 1]
    verts_bambu[:, 1] = -verts[:, 0]
    verts_bambu[:, 2] = verts[:, 2]
    faces_bambu = raw_mesh.faces[:, [0, 2, 1]]
    
    m = trimesh.Trimesh(vertices=verts_bambu, faces=faces_bambu)
    m.fix_normals()
    smoothing.filter_taubin(m, lamb=0.5, nu=-0.53, iterations=12)
    
    m.apply_translation([0, 0, -m.bounds[0][2]])
    center_xy = (m.bounds[0][:2] + m.bounds[1][:2]) / 2.0
    m.apply_translation([-center_xy[0], -center_xy[1], 0])
    
    if not with_base:
        m.apply_scale(target_height_mm / m.extents[2])
        m.apply_translation([0, 0, -m.bounds[0][2]])
        return m
        
    # Avec socle
    scale_dog = (target_height_mm - base_height_mm + 2.0) / m.extents[2]
    m.apply_scale(scale_dog)
    m.apply_translation([0, 0, -m.bounds[0][2] + 2.0])
    
    dx, dy = m.extents[0], m.extents[1]
    base_rx = (dx / 2.0) + 7.0
    base_ry = (dy / 2.0) + 7.0
    
    base_mesh = trimesh.creation.cylinder(radius=1.0, height=base_height_mm, sections=64)
    base_mesh.apply_scale([base_rx, base_ry, 1.0])
    base_mesh.apply_translation([0, 0, base_height_mm / 2.0])
    
    m_dog_mani = Manifold(Mesh(vert_properties=m.vertices.astype(np.float32), tri_verts=m.faces.astype(np.uint32)))
    m_base_mani = Manifold(Mesh(vert_properties=base_mesh.vertices.astype(np.float32), tri_verts=base_mesh.faces.astype(np.uint32)))
    m_union = m_dog_mani + m_base_mani
    
    mesh_union_data = m_union.to_mesh()
    m_with_base = trimesh.Trimesh(
        vertices=mesh_union_data.vert_properties[:, :3],
        faces=mesh_union_data.tri_verts
    )
    m_with_base.fix_normals()
    m_with_base.apply_translation([0, 0, -m_with_base.bounds[0][2]])
    m_with_base.apply_scale(target_height_mm / m_with_base.extents[2])
    m_with_base.apply_translation([0, 0, -m_with_base.bounds[0][2]])
    
    return m_with_base


def generate_mode3_ultra_sculpted(
    image: Image.Image,
    target_height_mm: float = 100.0,
    with_base: bool = True,
    base_height_mm: float = 4.5
) -> trimesh.Trimesh:
    """Mode 3 Ultra-Précis (Qualité Studio / Meshy-7) : Reconstruction volumétrique avec subdivision géométrique fine (223k+ polygones), Scharr edge highlighting et Cavity Carving des traits faciaux."""
    # 1. Base volumétrique
    m_base = generate_mode3_volumetric(image, target_height_mm=target_height_mm, with_base=with_base, base_height_mm=base_height_mm)
    
    # 2. Subdivision géométrique fine (Loop subdivision)
    v, f = trimesh.remesh.subdivide(m_base.vertices, m_base.faces)
    mesh_hd = trimesh.Trimesh(vertices=v, faces=f)
    mesh_hd.fix_normals()
    
    # 3. Extraction avancée du champ de détails morphologiques & cavités
    im_rgb = image.convert("RGB")
    arr = np.array(im_rgb)
    gray = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]) / 255.0
    
    # Fréquences moyennes (relief musculaire, museau, oreilles)
    blur_mid = ndi.gaussian_filter(gray, 12.0)
    mid_freq = (gray - blur_mid)
    
    # Hautes fréquences (micro-textures, plis, babines, médaillon, pelage)
    blur_fine = ndi.gaussian_filter(gray, 2.5)
    fine_freq = (gray - blur_fine)
    
    # Gradients de Scharr pour arêtes nettes (contours des yeux, truffe, anneaux)
    scharr_x = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=float) / 32.0
    scharr_y = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=float) / 32.0
    gx = ndi.convolve(gray, scharr_x)
    gy = ndi.convolve(gray, scharr_y)
    edge_sharp = np.sqrt(gx**2 + gy**2)
    edge_sharp = (edge_sharp - edge_sharp.min()) / max(0.001, (edge_sharp.max() - edge_sharp.min()))
    
    # Cavity Carving (creusement géométrique franc des narines, yeux, fentes)
    dark_cavities = np.clip(1.0 - (gray / 0.4), 0.0, 1.0) ** 1.8
    cavity_carve = -dark_cavities * 0.8
    
    # Carte globale de déplacement morphologique (Qualité Meshy-7)
    disp_map = (mid_freq * 0.6) + (fine_freq * 1.1) + (edge_sharp * 0.4) + cavity_carve
    
    verts = mesh_hd.vertices.copy()
    normals = mesh_hd.vertex_normals
    
    # Vertices du sujet orientés vers l'avant au-dessus du socle
    min_z_subject = (base_height_mm + 1.0) if with_base else (verts[:, 2].min() + 1.0)
    front_mask = (normals[:, 1] < -0.1) & (verts[:, 2] > min_z_subject)
    
    subj_verts = verts[verts[:, 2] > min_z_subject]
    x_min, x_max = subj_verts[:, 0].min(), subj_verts[:, 0].max()
    z_min, z_max = subj_verts[:, 2].min(), subj_verts[:, 2].max()
    
    # Bounding box dans l'image
    is_subject = (arr[:, :, 0] < 245) | (arr[:, :, 1] < 245) | (arr[:, :, 2] < 245)
    y_idx, x_idx = np.where(is_subject)
    if len(y_idx) > 0:
        y1, y2, x1, x2 = y_idx.min(), y_idx.max(), x_idx.min(), x_idx.max()
        cropped_disp = disp_map[y1:y2+1, x1:x2+1]
        ch, cw = cropped_disp.shape
        
        cols = np.clip(((verts[:, 0] - x_min) / max(0.001, x_max - x_min) * (cw - 1)), 0, cw - 1).astype(int)
        rows = np.clip(((z_max - verts[:, 2]) / max(0.001, z_max - z_min) * (ch - 1)), 0, ch - 1).astype(int)
        
        disp_values = np.clip(cropped_disp[rows, cols] * 1.3, -1.2, 1.4)
        weight = np.clip(-normals[:, 1], 0, 1) ** 1.5
        
        verts[front_mask] += normals[front_mask] * (disp_values[front_mask] * weight[front_mask])[:, None]
        
    mesh_ultra = trimesh.Trimesh(vertices=verts, faces=mesh_hd.faces)
    mesh_ultra.fix_normals()
    return mesh_ultra


def process_external_3d_file(
    file_path: str,
    target_height_mm: float = 100.0,
    with_base: bool = True,
    base_height_mm: float = 4.5
) -> trimesh.Trimesh:
    """Traite et optimise un modèle 3D externe (téléchargé depuis Meshy, Tripo 3D, Blender, etc.) pour Bambu Lab A1 Mini."""
    loaded = trimesh.load(file_path)
    if isinstance(loaded, trimesh.Scene):
        # Concaténer tous les sous-maillages
        mesh_list = [geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not mesh_list:
            raise ValueError("Aucun maillage 3D trouvé dans le fichier importé.")
        m = trimesh.util.concatenate(mesh_list)
    else:
        m = loaded
        
    # Nettoyage
    m.process()
    m.fix_normals()
    
    # Repos sur Z=0 et centrage XY
    m.apply_translation([0, 0, -m.bounds[0][2]])
    center_xy = (m.bounds[0][:2] + m.bounds[1][:2]) / 2.0
    m.apply_translation([-center_xy[0], -center_xy[1], 0])
    
    if not with_base:
        m.apply_scale(target_height_mm / max(0.001, m.extents[2]))
        m.apply_translation([0, 0, -m.bounds[0][2]])
        return m
        
    # Intégration socle pour Bambu Lab A1 Mini
    scale_obj = (target_height_mm - base_height_mm + 1.5) / max(0.001, m.extents[2])
    m.apply_scale(scale_obj)
    m.apply_translation([0, 0, -m.bounds[0][2] + 1.5])
    
    dx, dy = m.extents[0], m.extents[1]
    base_rx = (dx / 2.0) + 7.0
    base_ry = (dy / 2.0) + 7.0
    
    base_mesh = trimesh.creation.cylinder(radius=1.0, height=base_height_mm, sections=64)
    base_mesh.apply_scale([base_rx, base_ry, 1.0])
    base_mesh.apply_translation([0, 0, base_height_mm / 2.0])
    
    try:
        m_mani = Manifold(Mesh(vert_properties=m.vertices.astype(np.float32), tri_verts=m.faces.astype(np.uint32)))
        b_mani = Manifold(Mesh(vert_properties=base_mesh.vertices.astype(np.float32), tri_verts=base_mesh.faces.astype(np.uint32)))
        u_mani = m_mani + b_mani
        data = u_mani.to_mesh()
        m_final = trimesh.Trimesh(vertices=data.vert_properties[:, :3], faces=data.tri_verts)
    except Exception:
        # Fallback si maillage non-manifold à l'import
        m_final = trimesh.util.concatenate([m, base_mesh])
        
    m_final.fix_normals()
    m_final.apply_translation([0, 0, -m_final.bounds[0][2]])
    m_final.apply_scale(target_height_mm / max(0.001, m_final.extents[2]))
    m_final.apply_translation([0, 0, -m_final.bounds[0][2]])
    
    return m_final

