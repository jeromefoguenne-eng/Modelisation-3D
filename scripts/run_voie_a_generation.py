"""
Script de génération Voie A :
1. Appel Stability AI Stable Fast 3D (SF3D)
2. Appel Tencent Hunyuan3D-2 Multi-View Shape Generation
"""

import os
import sys
import time
import shutil
from gradio_client import Client, handle_file
import trimesh

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.studio_renderer import render_mesh_preview

BRAIN_DIR = r"C:\Users\TrendingPC\.gemini\antigravity\brain\6f356426-1b48-4c89-8e8b-a62d57a95f48"
OUT_DIR = os.path.join(BASE_DIR, "outputs", "voie_a_sota_ai")
os.makedirs(OUT_DIR, exist_ok=True)

def run_voie_a():
    print("=" * 80)
    print("DEMARRAGE VOIE A : GENERATION 3D MULTI-VUES PAR IA NATIVE SOTA")
    print("=" * 80)

    token_path = os.path.join(BASE_DIR, "hf_token.txt")
    with open(token_path) as f:
        tok = f.read().strip()

    src_img = os.path.join(BASE_DIR, "uploads", "upload_019cc412_van_aert_original.png")
    
    # --- 1. GENERATION SF3D (Stable Fast 3D) ---
    print("\n[1/2] Lancement de Stable Fast 3D (Stability AI)...")
    try:
        t0 = time.time()
        client_sf3d = Client("stabilityai/stable-fast-3d", token=tok)
        res_sf3d = client_sf3d.predict(
            input_image=handle_file(src_img),
            foreground_ratio=0.88,
            remesh_option="None",
            vertex_count=-1,
            texture_size=2048,
            api_name="/run_button"
        )
        print(f"      SF3D Réponse reçue en {time.time()-t0:.1f}s : {res_sf3d}")
        # res_sf3d est (preview_img, 3d_model_path)
        glb_sf3d_raw = res_sf3d[1] if isinstance(res_sf3d, (list, tuple)) else res_sf3d
        out_sf3d = os.path.join(OUT_DIR, "vanaert_sf3d_native.glb")
        shutil.copy2(glb_sf3d_raw, out_sf3d)
        print(f"      SF3D Modèle sauvegardé : {out_sf3d} ({os.path.getsize(out_sf3d)/(1024*1024):.1f} Mo)")

        # Rendu studio SF3D
        m_sf3d = trimesh.load(out_sf3d, force="mesh")
        p_sf3d_front = os.path.join(OUT_DIR, "sf3d_preview_front.png")
        p_sf3d_34 = os.path.join(OUT_DIR, "sf3d_preview_34.png")
        render_mesh_preview(m_sf3d, p_sf3d_front, img_size=1000, angle_deg=0.0)
        render_mesh_preview(m_sf3d, p_sf3d_34, img_size=1000, angle_deg=25.0)
        shutil.copy2(p_sf3d_front, os.path.join(BRAIN_DIR, "comparatif_voie_a_sf3d_front.png"))
        shutil.copy2(p_sf3d_34, os.path.join(BRAIN_DIR, "comparatif_voie_a_sf3d_34.png"))
        print("      Aperçus SF3D générés avec succès !")
    except Exception as e:
        print(f"      Erreur SF3D : {e}")

    # --- 2. GENERATION HUNYUAN3D-2 ---
    print("\n[2/2] Lancement de Tencent Hunyuan3D-2 (Shape Generation)...")
    try:
        t0 = time.time()
        client_hy = Client("tencent/Hunyuan3D-2", token=tok)
        res_hy = client_hy.predict(
            caption="Wout van Aert winning Tour de France stage with raised index finger in victory yell, Visma Lease a Bike yellow jersey, Oakley cycling sunglasses, aero helmet, Cervelo bike, detailed high resolution 3D sculpture",
            image=handle_file(src_img),
            mv_image_front=None,
            mv_image_back=None,
            mv_image_left=None,
            mv_image_right=None,
            steps=32,
            guidance_scale=5.5,
            seed=42,
            octree_resolution=384,
            check_box_rembg=True,
            num_chunks=8000,
            randomize_seed=False,
            api_name="/shape_generation"
        )
        print(f"      Hunyuan3D Réponse reçue en {time.time()-t0:.1f}s : {res_hy}")
        glb_hy_raw = None
        if isinstance(res_hy, (list, tuple)):
            for item in res_hy:
                if isinstance(item, str) and item.endswith(('.glb', '.gltf', '.obj')):
                    glb_hy_raw = item
                    break
        elif isinstance(res_hy, str) and res_hy.endswith(('.glb', '.gltf', '.obj')):
            glb_hy_raw = res_hy

        if glb_hy_raw and os.path.exists(glb_hy_raw):
            out_hy = os.path.join(OUT_DIR, "vanaert_hunyuan3d_native.glb")
            shutil.copy2(glb_hy_raw, out_hy)
            print(f"      Hunyuan3D Modèle sauvegardé : {out_hy} ({os.path.getsize(out_hy)/(1024*1024):.1f} Mo)")
            m_hy = trimesh.load(out_hy, force="mesh")
            p_hy_front = os.path.join(OUT_DIR, "hunyuan_preview_front.png")
            p_hy_34 = os.path.join(OUT_DIR, "hunyuan_preview_34.png")
            render_mesh_preview(m_hy, p_hy_front, img_size=1000, angle_deg=0.0)
            render_mesh_preview(m_hy, p_hy_34, img_size=1000, angle_deg=25.0)
            shutil.copy2(p_hy_front, os.path.join(BRAIN_DIR, "comparatif_voie_a_hunyuan_front.png"))
            shutil.copy2(p_hy_34, os.path.join(BRAIN_DIR, "comparatif_voie_a_hunyuan_34.png"))
            print("      Aperçus Hunyuan3D générés avec succès !")
    except Exception as e:
        print(f"      Erreur Hunyuan3D : {e}")

    print("\n" + "=" * 80)
    print("VOIE A TERMINEE !")
    print("=" * 80)

if __name__ == "__main__":
    run_voie_a()
