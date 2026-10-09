"""
Agent Antigravity — PNG to 3D Print Web Application
FastAPI Server backend pour la modélisation 3D et l'impression Bambu Lab A1 Mini.
"""

import os
import io
import uuid
import shutil
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
import trimesh

import engine

app = FastAPI(title="Agent Antigravity — PNG to 3D Print", version="1.0.0")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


@app.get("/api/presets")
def get_presets():
    """Préréglages d'impression optimisés pour la Bambu Lab A1 Mini."""
    return [
        {
            "id": "statuette_volumetrique_10cm",
            "name": "Statuette Volumétrique 3D (10 cm)",
            "description": "Objet 3D complet à 360° avec socle solide pour adhérence maximale sur A1 Mini.",
            "mode": 3,
            "target_height_mm": 100.0,
            "base_thickness_mm": 4.5,
            "relief_height_mm": 95.5,
            "add_base": True,
            "invert": False,
            "resolution": 256,
            "infill_recommendation": "12% Gyroid",
            "supports": "Arborescents (Tree Supports)"
        },
        {
            "id": "relief_decoratif",
            "name": "Relief Décoratif Mural",
            "description": "Plaque plate avec sujet mis en valeur en haut-relief (support mural ou présentoir).",
            "mode": 1,
            "target_height_mm": 100.0,
            "base_thickness_mm": 3.0,
            "relief_height_mm": 4.0,
            "add_base": True,
            "invert": False,
            "resolution": 250,
            "infill_recommendation": "15% Grille",
            "supports": "Aucun"
        },
        {
            "id": "medaille_badge",
            "name": "Médaille ou Badge",
            "description": "Format compact et fin pour trophée, porte-clé ou insigne.",
            "mode": 1,
            "target_height_mm": 60.0,
            "base_thickness_mm": 2.5,
            "relief_height_mm": 1.5,
            "add_base": True,
            "invert": False,
            "resolution": 250,
            "infill_recommendation": "20% Rectiligne",
            "supports": "Aucun"
        },
        {
            "id": "pion_jeu",
            "name": "Pion de Jeu de Société",
            "description": "Figurine robuste sur socle épais stable pour manipulation répétée.",
            "mode": 1,
            "target_height_mm": 45.0,
            "base_thickness_mm": 4.0,
            "relief_height_mm": 5.0,
            "add_base": True,
            "invert": False,
            "resolution": 250,
            "infill_recommendation": "25% Gyroid",
            "supports": "Aucun"
        },
        {
            "id": "plaque_logo",
            "name": "Plaque avec Logo",
            "description": "Idéal pour marquage commercial, enseigne de bureau ou signalétique.",
            "mode": 1,
            "target_height_mm": 120.0,
            "base_thickness_mm": 2.5,
            "relief_height_mm": 3.0,
            "add_base": True,
            "invert": False,
            "resolution": 280,
            "infill_recommendation": "15% Grille",
            "supports": "Aucun"
        },
        {
            "id": "photo_lithophane",
            "name": "Photo en Relief (Lithophane)",
            "description": "Traduction continue des nuances de gris en hauteurs variables.",
            "mode": 2,
            "target_height_mm": 100.0,
            "base_thickness_mm": 1.2,
            "relief_height_mm": 3.2,
            "add_base": True,
            "invert": False,
            "resolution": 250,
            "infill_recommendation": "100% Solide (pour rétroéclairage)",
            "supports": "Aucun"
        }
    ]


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload et analyse immédiate d'une image."""
    file_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".png", ".jpg", ".jpeg"]:
        raise HTTPException(status_code=400, detail="Format non supporté (seuls PNG et JPG sont acceptés).")
        
    filename = f"{file_id}_{file.filename}"
    save_path = os.path.join(UPLOAD_DIR, filename)
    
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    im = Image.open(save_path)
    w, h = im.size
    has_alpha = im.mode == "RGBA" and any(im.getchannel("A").getextrema()[0] < 255 if "A" in im.getbands() else False)
    
    # Aperçu nettoyé
    clean_im = engine.process_image_cleanup(im, remove_bg=True)
    clean_filename = f"clean_{filename}"
    clean_path = os.path.join(UPLOAD_DIR, clean_filename)
    clean_im.save(clean_path)
    
    return {
        "filename": filename,
        "clean_filename": clean_filename,
        "original_width": w,
        "original_height": h,
        "mode": im.mode,
        "has_transparent_bg": has_alpha,
        "preview_url": f"/static/uploads/{clean_filename}"
    }


@app.post("/api/convert")
async def convert_image(
    filename: str = Form(...),
    mode: int = Form(1),
    engine_mode: str = Form("ultra"), # "ultra" (Meshy-Grade 223k) or "standard"
    target_height_mm: float = Form(100.0),
    base_thickness_mm: float = Form(3.0),
    relief_height_mm: float = Form(3.0),
    add_base: bool = Form(True),
    invert: bool = Form(False),
    resolution: int = Form(250),
    smooth_sigma: float = Form(1.0)
):
    """Point d'entrée principal de génération 3D selon le mode choisi."""
    img_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(img_path):
        # Chercher dans static/demo si image de démo
        demo_path = os.path.join(STATIC_DIR, "demo", filename)
        if os.path.exists(demo_path):
            img_path = demo_path
        else:
            raise HTTPException(status_code=404, detail="Image introuvable.")
            
    im = Image.open(img_path)
    out_id = str(uuid.uuid4())[:8]
    
    try:
        if mode == 1:
            mesh = engine.generate_mode1_extruded(
                image=im,
                target_height_mm=target_height_mm,
                base_thickness_mm=base_thickness_mm,
                relief_height_mm=relief_height_mm,
                add_base=add_base,
                invert=invert
            )
            mode_name = "Mode1_Extrusion"
        elif mode == 2:
            mesh = engine.generate_mode2_heightmap(
                image=im,
                target_height_mm=target_height_mm,
                base_thickness_mm=base_thickness_mm,
                max_relief_mm=relief_height_mm,
                invert=invert,
                resolution=resolution,
                smooth_sigma=smooth_sigma
            )
            mode_name = "Mode2_Lithophane"
        elif mode == 3:
            if engine_mode == "ultra":
                mesh = engine.generate_mode3_ultra_sculpted(
                    image=im,
                    target_height_mm=target_height_mm,
                    with_base=add_base,
                    base_height_mm=base_thickness_mm
                )
                mode_name = "Mode3_Studio_Ultra_HD"
            else:
                mesh = engine.generate_mode3_volumetric(
                    image=im,
                    target_height_mm=target_height_mm,
                    with_base=add_base,
                    base_height_mm=base_thickness_mm,
                    resolution=256
                )
                mode_name = "Mode3_Volumetrique3D"
        else:
            raise HTTPException(status_code=400, detail="Mode inconnu.")
            
        stl_name = f"{mode_name}_{out_id}.stl"
        threemf_name = f"{mode_name}_{out_id}.3mf"
        
        stl_path = os.path.join(OUTPUT_DIR, stl_name)
        threemf_path = os.path.join(OUTPUT_DIR, threemf_name)
        
        mesh.export(stl_path)
        mesh.export(threemf_path)
        
        printability = engine.check_bambu_printability(mesh)
        
        return {
            "success": True,
            "stl_filename": stl_name,
            "threemf_filename": threemf_name,
            "stl_url": f"/outputs/{stl_name}",
            "threemf_url": f"/outputs/{threemf_name}",
            "printability": printability
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/import-3d")
async def import_external_3d(
    file: UploadFile = File(...),
    target_height_mm: float = Form(100.0),
    with_base: bool = Form(True),
    base_height_mm: float = Form(4.5)
):
    """Importe et convertit directement un modèle 3D externe (issu de Meshy.ai, Tripo3D.ai, etc.) pour Bambu Lab A1 Mini."""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".glb", ".gltf", ".obj", ".stl"]:
        raise HTTPException(status_code=400, detail="Format 3D non supporté (.glb, .gltf, .obj, .stl uniquement).")
        
    file_id = str(uuid.uuid4())[:8]
    in_name = f"import_{file_id}_{file.filename}"
    in_path = os.path.join(UPLOAD_DIR, in_name)
    
    with open(in_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        mesh = engine.process_external_3d_file(
            file_path=in_path,
            target_height_mm=target_height_mm,
            with_base=with_base,
            base_height_mm=base_height_mm
        )
        
        out_stl = f"Meshy_Import_A1Mini_{file_id}.stl"
        out_3mf = f"Meshy_Import_A1Mini_{file_id}.3mf"
        
        stl_path = os.path.join(OUTPUT_DIR, out_stl)
        threemf_path = os.path.join(OUTPUT_DIR, out_3mf)
        
        mesh.export(stl_path)
        mesh.export(threemf_path)
        
        printability = engine.check_bambu_printability(mesh)
        
        return {
            "success": True,
            "stl_filename": out_stl,
            "threemf_filename": out_3mf,
            "stl_url": f"/outputs/{out_stl}",
            "threemf_url": f"/outputs/{out_3mf}",
            "printability": printability
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erreur de traitement du modèle 3D : {str(e)}")


@app.get("/api/demo-models")
def get_demo_models():
    """Retourne les modèles 3D précalculés haute fidélité (dont Babel Ultra-HD 10 cm)."""
    return [
        {
            "id": "babel_ultra_detail_10cm",
            "name": "Babel - Chien Ultra-Précis Studio (10 cm)",
            "description": "Reconstruction volumétrique 360° avec micro-sculpture faciale (yeux, truffe, babines, médaillon) — Qualité Meshy / Studio.",
            "stl_url": "/outputs/Babel_Ultra_Detail_MeshyGrade_10cm.stl",
            "threemf_url": "/outputs/Babel_Ultra_Detail_MeshyGrade_10cm.3mf",
            "preview_image": "/static/demo/sans_titre.png",
            "image_url": "/static/demo/sans_titre.png",
            "height_mm": 100.0,
            "width_mm": 119.0,
            "depth_mm": 114.6,
            "mode": 3,
            "is_watertight": True,
            "vertices": 223944,
            "faces": 447888,
            "estimated_weight_g": 39.5,
            "estimated_time": "1h 50min"
        },
        {
            "id": "babel_sans_titre_10cm",
            "name": "Babel - Chien Standard Lissé (10 cm)",
            "description": "Reconstruction 3D volumétrique TripoSR standard avec socle pour Bambu Lab A1 Mini.",
            "stl_url": "/outputs/Babel_Sans_Titre_Avec_Socle_10cm.stl",
            "threemf_url": "/outputs/Babel_Sans_Titre_Avec_Socle_10cm.3mf",
            "preview_image": "/static/demo/sans_titre.png",
            "image_url": "/static/demo/sans_titre.png",
            "height_mm": 100.0,
            "width_mm": 119.0,
            "depth_mm": 114.6,
            "mode": 3,
            "is_watertight": True,
            "vertices": 44449,
            "faces": 88890,
            "estimated_weight_g": 38.5,
            "estimated_time": "1h 45min"
        },
        {
            "id": "babel_sans_titre_pure_10cm",
            "name": "Babel - Sculpture Pure Sans Socle (10 cm)",
            "description": "Sculpture pure du chien autonome à 360°, hauteur 100 mm.",
            "stl_url": "/outputs/Babel_Sans_Titre_Pure_10cm.stl",
            "threemf_url": "/outputs/Babel_Sans_Titre_Pure_10cm.3mf",
            "preview_image": "/static/demo/sans_titre.png",
            "image_url": "/static/demo/sans_titre.png",
            "height_mm": 100.0,
            "width_mm": 107.5,
            "depth_mm": 103.0,
            "mode": 3,
            "is_watertight": True,
            "vertices": 44411,
            "faces": 88810,
            "estimated_weight_g": 32.0,
            "estimated_time": "1h 25min"
        }
    ]



@app.get("/")
def serve_index():
    """Sert l'interface utilisateur web."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
