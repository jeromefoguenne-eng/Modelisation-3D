"""
Client d'intégration pour Tencent Hunyuan3D-2.1 Cloud via Hugging Face Gradio Client.
Permet d'obtenir des maillages 3D complets 360° générés par DiT (Diffusion Transformer).
"""

import os
import shutil
import trimesh
from gradio_client import Client, handle_file

HF_SPACE_ID = "tencent/Hunyuan3D-2.1"

def get_hf_token() -> str:
    """Récupère le token HF si configuré."""
    token_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "hf_token.txt")
    if os.path.exists(token_file):
        with open(token_file, "r") as f:
            token = f.read().strip()
            if len(token) > 10:
                return token
    return os.environ.get("HF_TOKEN", None)

def generate_hunyuan_mesh(
    image_rgba_path: str,
    output_dir: str,
    octree_resolution: int = 384,
    steps: int = 30
) -> trimesh.Trimesh:
    """
    Appelle l'API Tencent Hunyuan3D-2.1 pour générer un fichier GLB 3D complet.
    """
    token = get_hf_token()
    print(f"[HunyuanEngine] Connexion à l'espace {HF_SPACE_ID} (Token présent: {bool(token)})...")
    
    try:
        client = Client(HF_SPACE_ID, hf_token=token)
        
        # Appel de l'API de génération de forme
        # Note: L'espace Hunyuan3D-2.1 expose l'endpoint /generation_all ou /shape_generation
        print(f"[HunyuanEngine] Envoi de l'image {image_rgba_path}...")
        
        # Tentative d'appel standard
        result = client.predict(
            image=handle_file(image_rgba_path),
            api_name="/generation_all"
        )
        
        # Le résultat est généralement un tuple contenant le chemin du fichier .glb
        glb_path = None
        if isinstance(result, (list, tuple)):
            for item in result:
                if isinstance(item, str) and item.endswith(('.glb', '.gltf', '.obj')):
                    glb_path = item
                    break
        elif isinstance(result, str) and result.endswith(('.glb', '.gltf', '.obj')):
            glb_path = result
            
        if not glb_path or not os.path.exists(glb_path):
            raise RuntimeError(f"Fichier 3D non trouvé dans la réponse Hunyuan: {result}")
            
        print(f"[HunyuanEngine] Modèle reçu : {glb_path}")
        
        # Chargement Trimesh
        mesh = trimesh.load(glb_path, force='mesh')
        return mesh

    except Exception as e:
        print(f"[HunyuanEngine] Échec de la génération Cloud Hunyuan: {e}")
        raise
