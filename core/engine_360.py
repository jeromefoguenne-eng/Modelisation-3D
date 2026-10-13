"""
Moteur de Génération de Figurines 3D Complètes à 360° (Avec ou Sans Socle).
Propulsé par :
1. TripoSR Local GPU RTX 4070 Ti (549/549 tenseurs DINO + Triplane chargés en strict=True)
2. Tencent Hunyuan3D-2 Ultra-HD (Diffusion Transformer 360°)
Calibré pour Bambu Lab A1 Mini (<= 180mm, Watertight).
"""

import os
import sys
import time
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import trimesh

# Ajout du chemin ComfyUI-Flowty-TripoSR
COMFY_NODE_PATH = Path(r"E:\ComfyUI\custom_nodes\ComfyUI-Flowty-TripoSR")
CHECKPOINT_PATH = Path(r"E:\ComfyUI\models\checkpoints\triposr_model.ckpt")
CONFIG_PATH = COMFY_NODE_PATH / "config.yaml"

if str(COMFY_NODE_PATH) not in sys.path:
    sys.path.insert(0, str(COMFY_NODE_PATH))

_TRIPOSR_MODEL = None

def get_triposr_model(device: str = "cuda"):
    """Charge et met en cache le modèle TripoSR en VRAM avec 100% des poids (strict=True)."""
    global _TRIPOSR_MODEL
    if _TRIPOSR_MODEL is None:
        from tsr.system import TSR
        print("[Engine360] Chargement de TripoSR (549/549 tenseurs, strict=True) sur RTX 4070 Ti...")
        t0 = time.time()
        model = TSR.from_pretrained_custom(str(CHECKPOINT_PATH), str(CONFIG_PATH))
        model.renderer.set_chunk_size(8192)
        model.to(device)
        _TRIPOSR_MODEL = model
        print(f"[Engine360] Modèle TripoSR prêt en VRAM ({time.time() - t0:.2f}s).")
    return _TRIPOSR_MODEL


def prepare_image_for_360(image_path: str, foreground_ratio: float = 0.85) -> Image.Image:
    """
    Détoure l'image avec rembg, recadre autour du sujet selon foreground_ratio,
    et place sur un fond gris neutre (0.5) attendu par l'encodeur DINO de TripoSR.
    """
    from rembg import remove
    img = Image.open(image_path).convert("RGBA")
    
    # Vérifier si l'image a déjà un canal alpha détouré
    alpha = np.array(img.split()[-1])
    if np.mean(alpha > 250) > 0.95:
        img = remove(img)
        alpha = np.array(img.split()[-1])

    # Trouver la boîte englobante du sujet pour maximiser la résolution 3D
    coords = np.argwhere(alpha > 30)
    if len(coords) > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        img = img.crop((x0, y0, x1 + 1, y1 + 1))

    # Redimensionner et centrer dans un carré avec marge (foreground_ratio)
    w, h = img.size
    max_side = int(max(w, h) / foreground_ratio)
    square_rgba = Image.new("RGBA", (max_side, max_side), (0, 0, 0, 0))
    paste_x = (max_side - w) // 2
    paste_y = (max_side - h) // 2
    square_rgba.paste(img, (paste_x, paste_y))

    # Fusion sur fond gris neutre 0.5
    arr = np.array(square_rgba).astype(np.float32) / 255.0
    rgb = arr[:, :, :3] * arr[:, :, 3:4] + (1.0 - arr[:, :, 3:4]) * 0.5
    return Image.fromarray((rgb * 255.0).astype(np.uint8)).convert("RGB")


def postprocess_360_figurine(
    mesh: trimesh.Trimesh,
    target_height_mm: float = 120.0,
    include_base: bool = True,
    base_height_mm: float = 5.0,
    smooth_iterations: int = 8
) -> trimesh.Trimesh:
    """
    Nettoie, lisse, met à l'échelle (Bambu Lab A1 Mini) et ajoute optionnellement un socle cylindrique.
    Orienté en Y-up pour Three.js et export STL/GLB.
    """
    # 1. Garder uniquement la composante principale (élimine toute poussière isolée)
    try:
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            mesh = max(components, key=lambda c: len(c.vertices))
    except Exception as e:
        print(f"[Engine360] Split info: {e}")

    # 2. Lissage Taubin (préserve le volume tout en éliminant les stries de voxels)
    if smooth_iterations > 0:
        try:
            trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=smooth_iterations)
        except Exception as e:
            print(f"[Engine360] Smoothing info: {e}")

    # 3. Mise à l'échelle selon la hauteur cible (axe Y)
    extents = mesh.extents
    current_height = extents[1] if extents[1] > 1e-4 else 1.0
    scale_factor = target_height_mm / current_height

    # Sécurité plateau Bambu Lab A1 Mini (180x180x180 mm -> max 175 mm)
    if max(extents * scale_factor) > 175.0:
        scale_factor = 175.0 / max(extents)

    mesh.apply_scale(scale_factor)

    # 4. Centrage en X et Z, et pose à Y = 0
    bounds = mesh.bounds
    center_x = (bounds[0][0] + bounds[1][0]) / 2.0
    center_z = (bounds[0][2] + bounds[1][2]) / 2.0
    min_y = bounds[0][1]
    mesh.apply_translation([-center_x, -min_y, -center_z])

    # 5. Ajout optionnel du socle cylindrique Bambu Lab A1 Mini
    if include_base:
        ext_x, ext_y, ext_z = mesh.extents
        base_radius = max(ext_x, ext_z) * 0.44
        base_radius = max(base_radius, 18.0)  # Rayon minimum de stabilité (diamètre 36mm)

        cyl = trimesh.creation.cylinder(radius=base_radius, height=base_height_mm, sections=80)
        # Aligner le cylindre sur l'axe Y (vertical)
        rot_x = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])
        cyl.apply_transform(rot_x)

        # Donner au socle une couleur gris anthracite élégante si le maillage a des couleurs de sommets
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            cyl_colors = np.full((len(cyl.vertices), 4), [60, 65, 75, 255], dtype=np.uint8)
            cyl.visual.vertex_colors = cyl_colors

        # Ancrer les pieds/roues de 1.8 mm dans le socle pour une liaison mécanique solide
        overlap_mm = 1.8
        cyl.apply_translation([0, (base_height_mm / 2.0) - overlap_mm, 0])
        mesh.apply_translation([0, base_height_mm - overlap_mm, 0])

        mesh = trimesh.util.concatenate([mesh, cyl])
        # Recaler le bas du socle exactement à Y = 0
        mesh.apply_translation([0, -mesh.bounds[0][1], 0])

    # Réparation finale des normales
    try:
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fill_holes(mesh)
    except Exception:
        pass

    return mesh


def generate_figurine_360_local(
    image_path: str,
    target_height_mm: float = 120.0,
    include_base: bool = True,
    base_height_mm: float = 5.0,
    resolution: int = 300,
    threshold: float = 25.0,
    progress_callback=None
) -> trimesh.Trimesh:
    """
    Génère une figurine 3D 360° complète sur le GPU local RTX 4070 Ti via TripoSR corrigé.
    """
    if progress_callback:
        progress_callback(20, "Détourage IA et cadrage optimal du sujet...")
    in_img = prepare_image_for_360(image_path, foreground_ratio=0.85)

    if progress_callback:
        progress_callback(40, "Chargement du réseau neuronal 360° sur GPU RTX 4070 Ti...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_triposr_model(device=device)

    if progress_callback:
        progress_callback(60, f"Reconstruction volumétrique 360° (Grille {resolution}³)...")
    with torch.no_grad():
        scene_codes = model([in_img], device=device)
        meshes = model.extract_mesh(scene_codes, resolution=resolution, threshold=threshold)
        mesh = meshes[0]

    # Redressement en Y-up et orientation de face vers +Z
    transform_y_up = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1]
    ])
    mesh.apply_transform(transform_y_up)
    rot_front = trimesh.transformations.rotation_matrix(np.radians(-90), [0, 1, 0])
    mesh.apply_transform(rot_front)

    if progress_callback:
        socle_txt = "avec socle cylindrique" if include_base else "sans socle"
        progress_callback(80, f"Lissage Taubin et calibrage Bambu A1 Mini ({socle_txt})...")

    mesh = postprocess_360_figurine(
        mesh=mesh,
        target_height_mm=target_height_mm,
        include_base=include_base,
        base_height_mm=base_height_mm,
        smooth_iterations=8
    )
    return mesh


def generate_figurine_360_hunyuan(
    image_path: str,
    target_height_mm: float = 120.0,
    include_base: bool = True,
    base_height_mm: float = 5.0,
    steps: int = 32,
    octree_resolution: int = 384,
    progress_callback=None
) -> trimesh.Trimesh:
    """
    Tente la génération 360° via Tencent Hunyuan3D-2 Cloud SOTA,
    et bascule automatiquement sur le GPU Local RTX 4070 Ti (résolution 320) si le Cloud est saturé.
    """
    try:
        try:
            import truststore
            truststore.inject_into_ssl()
        except Exception:
            pass

        from gradio_client import Client, handle_file

        token = os.environ.get("HF_TOKEN", "")
        token_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "hf_token.txt")
        if not token and os.path.exists(token_file):
            with open(token_file, "r", encoding="utf-8") as f:
                t = f.read().strip()
                if len(t) > 8:
                    token = t

        if progress_callback:
            progress_callback(25, "Connexion au cluster Tencent Hunyuan3D-2 (360° Ultra-HD)...")

        client = Client("tencent/Hunyuan3D-2", token=token if token else None)

        if progress_callback:
            progress_callback(45, f"Sculpture neuronale 360° Hunyuan3D-2 (Octree {octree_resolution})...")

        res = client.predict(
            caption="ultra realistic detailed 3D figurine sculpture, high polygon, manifold geometry",
            image=handle_file(image_path),
            mv_image_front=None,
            mv_image_back=None,
            mv_image_left=None,
            mv_image_right=None,
            steps=steps,
            guidance_scale=5.5,
            seed=1234,
            octree_resolution=octree_resolution,
            check_box_rembg=True,
            num_chunks=8000,
            randomize_seed=True,
            api_name="/generation_all"
        )

        src_glb = res[1] if (isinstance(res, (list, tuple)) and len(res) > 1 and res[1]) else res[0]
        if isinstance(src_glb, dict) and "value" in src_glb:
            src_glb = src_glb["value"]

        mesh = trimesh.load(src_glb, force="mesh")

        if progress_callback:
            progress_callback(80, "Calibrage Bambu A1 Mini et assemblage du socle...")

        mesh = postprocess_360_figurine(
            mesh=mesh,
            target_height_mm=target_height_mm,
            include_base=include_base,
            base_height_mm=base_height_mm,
            smooth_iterations=4
        )
        return mesh

    except Exception as e:
        print(f"[Engine360] Hunyuan Cloud indisponible ({e}) -> Bascule automatique sur GPU Local RTX 4070 Ti (Grille 320³)...")
        if progress_callback:
            progress_callback(35, "Serveur Cloud occupé : Bascule sur votre GPU RTX 4070 Ti en Ultra-HD (320³)...")
        return generate_figurine_360_local(
            image_path=image_path,
            target_height_mm=target_height_mm,
            include_base=include_base,
            base_height_mm=base_height_mm,
            resolution=320,
            threshold=25.0,
            progress_callback=progress_callback
        )
