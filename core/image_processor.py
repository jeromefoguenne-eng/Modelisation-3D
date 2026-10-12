"""
Module de traitement d'images haute fidélité pour modélisation 3D.
Détourage IA via rembg, extraction de masques et rehaussement de contrastes.
"""

import os
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from rembg import remove, new_session

# Session u2net réutilisable pour de meilleures performances
_REMBG_SESSION = None

def get_rembg_session():
    global _REMBG_SESSION
    if _REMBG_SESSION is None:
        try:
            _REMBG_SESSION = new_session("u2net")
        except Exception as e:
            print(f"[ImageProcessor] Avertissement session u2net: {e}, fallback session par défaut")
            _REMBG_SESSION = None
    return _REMBG_SESSION

def process_input_image(input_path: str, output_rgba_path: str, max_size: int = 1024) -> dict:
    """
    Traite l'image source :
    1. Redimensionne avec ratio d'aspect préservé
    2. Détoure le sujet principal via rembg si pas déjà transparent
    3. Normalise le masque alpha et applique un padding de sécurité
    4. Retourne les métadonnées et chemins
    """
    img = Image.open(input_path)
    
    # Conversion initiale
    if img.mode not in ('RGB', 'RGBA'):
        img = img.convert('RGBA' if 'transparency' in img.info else 'RGB')
        
    # Redimensionnement si trop grand
    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    # Détection de transparence existante
    has_alpha = False
    if img.mode == 'RGBA':
        alpha_arr = np.array(img.split()[-1])
        if np.any(alpha_arr < 250):
            has_alpha = True
            
    if not has_alpha:
        print(f"[ImageProcessor] Détourage IA via rembg sur {input_path}...")
        session = get_rembg_session()
        if session:
            img_rgba = remove(img, session=session)
        else:
            img_rgba = remove(img)
    else:
        img_rgba = img

    # Nettoyage et padding de sécurité (5% sur les bords pour éviter les coupures aux parois)
    w, h = img_rgba.size
    pad_x = int(w * 0.05)
    pad_y = int(h * 0.05)
    
    padded_img = Image.new('RGBA', (w + 2 * pad_x, h + 2 * pad_y), (0, 0, 0, 0))
    padded_img.paste(img_rgba, (pad_x, pad_y))
    
    # Sauvegarde
    os.makedirs(os.path.dirname(output_rgba_path), exist_ok=True)
    padded_img.save(output_rgba_path, format="PNG")
    
    # Extraction masque et luminance
    alpha_chan = padded_img.split()[-1]
    mask_np = (np.array(alpha_chan) > 128).astype(np.uint8) * 255
    
    # Image en niveau de gris contrastée pour les détails photoniques
    gray = ImageOps.grayscale(padded_img.convert('RGB'))
    # Rehaussement local de contraste
    gray_contrasted = ImageOps.autocontrast(gray, cutoff=1)
    
    return {
        "rgba_path": output_rgba_path,
        "width": padded_img.width,
        "height": padded_img.height,
        "mask": mask_np,
        "grayscale": np.array(gray_contrasted, dtype=np.float32) / 255.0
    }
