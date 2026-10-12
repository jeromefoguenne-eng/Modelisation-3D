"""
Application Serveur FastAPI - Modélisation 3D Haute Précision pour Bambu Lab A1 Mini.
Architecture asynchrone complète, support GPU RTX 4070 Ti, file de tâches avec progression en temps réel.
"""

import os
import uuid
import time
import shutil
import asyncio
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch

from core.engine_lithophane import create_pure_relief_mesh
from core.bambu_validator import validate_and_scale_for_bambu, export_all_formats
from core.studio_renderer import render_mesh_preview

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app = FastAPI(title="Studio 3D Bambu Lab A1 Mini", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montage des dossiers statiques
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")
if os.path.exists(UPLOADS_DIR):
    app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")

# File de suivi des tâches
TASKS = {}

def get_gpu_info():
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    vram_total_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if cuda_avail else 0
    return {
        "cuda_available": cuda_avail,
        "device_name": device_name,
        "vram_gb": vram_total_gb,
        "torch_version": torch.__version__
    }

@app.get("/")
def index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return HTMLResponse("<h1>Studio 3D Bambu Lab A1 Mini</h1><p>Frontend en cours de chargement...</p>")

@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "gpu": get_gpu_info(),
        "tasks_in_memory": len(TASKS)
    }

@app.get("/api/demo-images")
def list_demo_images():
    """Renvoie les images de test disponibles."""
    demos = []
    # Recherche dans uploads et racine
    search_dirs = [UPLOADS_DIR, os.path.join(BASE_DIR, "demo")]
    for d in search_dirs:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    # Chemin public
                    rel = f"/uploads/{f}" if d == UPLOADS_DIR else f"/demo/{f}"
                    demos.append({
                        "name": f,
                        "url": rel,
                        "local_path": os.path.join(d, f)
                    })
    return {"demos": demos}

def background_generation_task(
    task_id: str,
    image_path: str,
    mode: str,
    target_width_mm: float,
    relief_depth_mm: float,
    curvature_deg: float
):
    try:
        task_dir = os.path.join(OUTPUTS_DIR, task_id)
        os.makedirs(task_dir, exist_ok=True)
        
        TASKS[task_id]["progress"] = 20
        mode_label = "Bas-Relief Sculpté" if mode == "bas_relief" else "Lithophanie Rétro-éclairée"
        TASKS[task_id]["step"] = f"Génération Haute Fidélité ({mode_label}) au pixel près..."
        
        # Génération directe pure sans déformation
        mesh = create_pure_relief_mesh(
            image_path=image_path,
            target_width_mm=target_width_mm,
            mode=mode,
            base_thickness_mm=2.5 if mode == "bas_relief" else 0.8,
            relief_depth_mm=relief_depth_mm,
            curvature_angle_deg=curvature_deg,
            resolution_dpi=450
        )
        
        # Étape 2 : Calibrage Bambu Lab A1 Mini
        TASKS[task_id]["progress"] = 65
        TASKS[task_id]["step"] = "Calibrage Bambu Lab A1 Mini (plateau 180mm, solidité étanche)..."
        metrics = validate_and_scale_for_bambu(
            mesh,
            target_height_mm=0,  # Le moteur gère déjà l'échelle exacte en conservant le ratio
            enforce_watertight=True
        )
        
        # Étape 4 : Exportation des formats 3D
        TASKS[task_id]["progress"] = 85
        TASKS[task_id]["step"] = "Exportation des formats d'impression (STL, GLB, OBJ)..."
        base_export = os.path.join(task_dir, "model_bambu")
        files_dict = export_all_formats(mesh, base_export)
        
        # Étape 5 : Rendu Studio de prévisualisation HD
        TASKS[task_id]["progress"] = 95
        TASKS[task_id]["step"] = "Génération du rendu Studio HD..."
        preview_path = os.path.join(task_dir, "preview_studio.png")
        render_mesh_preview(mesh, preview_path, img_size=512, angle_deg=25.0)
        
        # Succès final
        TASKS[task_id]["progress"] = 100
        TASKS[task_id]["status"] = "completed"
        TASKS[task_id]["step"] = "Modélisation 3D terminée avec succès !"
        TASKS[task_id]["result"] = {
            "task_id": task_id,
            "mode": mode,
            "metrics": metrics,
            "files": {
                "stl": f"/outputs/{task_id}/model_bambu.stl",
                "glb": f"/outputs/{task_id}/model_bambu.glb",
                "obj": f"/outputs/{task_id}/model_bambu.obj",
                "preview": f"/outputs/{task_id}/preview_studio.png",
                "rgba": f"/outputs/{task_id}/input_rgba.png"
            }
        }
        print(f"[App] Tâche {task_id} terminée avec succès.")

    except Exception as e:
        import traceback
        traceback.print_exc()
        TASKS[task_id]["status"] = "failed"
        TASKS[task_id]["error"] = str(e)
        TASKS[task_id]["step"] = f"Erreur : {str(e)}"

@app.post("/api/convert")
async def start_conversion(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    demo_image_path: Optional[str] = Form(None),
    mode: str = Form("bas_relief"),
    target_width_mm: float = Form(120.0),
    relief_depth_mm: float = Form(4.5),
    curvature_deg: float = Form(0.0)
):
    """
    Lance une tâche de modélisation 3D en arrière-plan.
    """
    task_id = str(uuid.uuid4())[:8]
    task_upload_dir = os.path.join(UPLOADS_DIR, f"task_{task_id}")
    os.makedirs(task_upload_dir, exist_ok=True)
    
    local_img_path = None
    
    if file and file.filename:
        local_img_path = os.path.join(task_upload_dir, file.filename)
        with open(local_img_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    elif demo_image_path:
        # Trouver l'image correspondante
        if os.path.exists(demo_image_path):
            local_img_path = demo_image_path
        else:
            possible = os.path.join(BASE_DIR, demo_image_path.lstrip("/\\"))
            if os.path.exists(possible):
                local_img_path = possible
            else:
                basename = os.path.basename(demo_image_path)
                candidate = os.path.join(UPLOADS_DIR, basename)
                if os.path.exists(candidate):
                    local_img_path = candidate
                else:
                    raise HTTPException(status_code=400, detail="Fichier image introuvable")
    else:
        raise HTTPException(status_code=400, detail="Aucun fichier ni image de démo fourni")
        
    TASKS[task_id] = {
        "id": task_id,
        "status": "processing",
        "progress": 5,
        "step": "Tâche enregistrée. Démarrage de la modélisation...",
        "created_at": time.time(),
        "result": None,
        "error": None
    }
    
    # Démarrage en tâche de fond (non bloquante)
    background_tasks.add_task(
        background_generation_task,
        task_id=task_id,
        image_path=local_img_path,
        mode=mode,
        target_width_mm=target_width_mm,
        relief_depth_mm=relief_depth_mm,
        curvature_deg=curvature_deg
    )
    
    return {
        "task_id": task_id,
        "status": "processing",
        "message": "Conversion 3D démarrée avec succès."
    }

@app.get("/api/task/{task_id}")
def get_task_status(task_id: str):
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Tâche introuvable")
    return TASKS[task_id]

@app.get("/api/download/{task_id}/{file_format}")
def download_model(task_id: str, file_format: str):
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Tâche introuvable")
        
    ext_map = {"stl": "stl", "glb": "glb", "obj": "obj"}
    if file_format.lower() not in ext_map:
        raise HTTPException(status_code=400, detail="Format non supporté (utilisez stl, glb ou obj)")
        
    ext = ext_map[file_format.lower()]
    file_path = os.path.join(OUTPUTS_DIR, task_id, f"model_bambu.{ext}")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Fichier {file_format.upper()} non trouvé pour cette tâche")
        
    return FileResponse(
        file_path,
        filename=f"bambu_a1_mini_{task_id}.{ext}",
        media_type="application/octet-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8080, reload=False)
