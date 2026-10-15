"""
Application Serveur FastAPI - Studio 3D Figurine 360° (Bambu Lab A1 Mini).
Génération de figurines 3D complètes à 360° (avec ou sans socle) sur GPU RTX 4070 Ti et Hunyuan3D-2.
"""

import os
import uuid
import time
import shutil
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch

from core.engine_360 import (
    generate_figurine_360_hunyuan,
    generate_figurine_360_hunyuan_local
)
from core.bambu_validator import validate_and_scale_for_bambu, export_all_formats
from core.studio_renderer import render_mesh_preview

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app = FastAPI(title="Studio 3D Figurine 360° - Bambu Lab A1 Mini", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SITE_PASSWORD = "fmttnhech"
AUTH_TOKEN_VALUE = "auth_token_fmttnhech_verified"
AUTH_COOKIE_NAME = "studio3d_session"

from pydantic import BaseModel

class LoginRequest(BaseModel):
    password: str

@app.post("/api/auth/login")
async def auth_login(req: LoginRequest, response: Response):
    if req.password == SITE_PASSWORD:
        response.set_cookie(
            key=AUTH_COOKIE_NAME,
            value=AUTH_TOKEN_VALUE,
            httponly=False,
            max_age=30 * 24 * 3600,
            samesite="lax"
        )
        return {"success": True, "token": AUTH_TOKEN_VALUE}
    raise HTTPException(status_code=401, detail="Mot de passe incorrect")

@app.get("/api/auth/check")
def auth_check(request: Request):
    token = request.headers.get("X-Site-Auth") or request.cookies.get(AUTH_COOKIE_NAME)
    return {"authenticated": token == AUTH_TOKEN_VALUE}

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")
if os.path.exists(UPLOADS_DIR):
    app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")

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
    return HTMLResponse("<h1>Studio 3D Figurine 360°</h1>")

@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "gpu": get_gpu_info(),
        "tasks_in_memory": len(TASKS)
    }

@app.get("/api/demo-images")
def list_demo_images():
    """Renvoie les images de test disponibles (sans Van Aert)."""
    demos = []
    seen_names = set()
    if os.path.exists(UPLOADS_DIR):
        for f in sorted(os.listdir(UPLOADS_DIR)):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                if any(kw in f.lower() for kw in ["van_aert", "vanaert", "van aert"]):
                    continue
                full_p = os.path.join(UPLOADS_DIR, f)
                if os.path.isfile(full_p):
                    demos.append({
                        "name": f,
                        "url": f"/uploads/{f}",
                        "local_path": full_p
                    })
                    seen_names.add(f)
    return {"demos": demos[:24]}


def background_generation_task(
    task_id: str,
    image_path: str,
    mode: str,
    target_height_mm: float,
    include_base: bool,
    base_height_mm: float,
    base_size_mm: float,
    base_shape: str,
    resolution: int,
    hf_token: Optional[str] = None
):
    try:
        task_dir = os.path.join(OUTPUTS_DIR, task_id)
        os.makedirs(task_dir, exist_ok=True)

        def update_progress(pct: int, step_msg: str):
            TASKS[task_id]["progress"] = pct
            TASKS[task_id]["step"] = step_msg

        is_local_gpu_available = os.path.exists(r"E:\ComfyUI") and not os.environ.get("SPACE_ID")

        if mode == "hunyuan_360":
            try:
                octree_res = max(256, min(512, int(resolution)))
                mesh = generate_figurine_360_hunyuan(
                    image_path=image_path,
                    target_height_mm=target_height_mm,
                    include_base=include_base,
                    base_height_mm=base_height_mm,
                    base_size_mm=base_size_mm,
                    base_shape=base_shape,
                    steps=45,
                    octree_resolution=octree_res,
                    hf_token=hf_token,
                    progress_callback=update_progress
                )
            except Exception as e_hy:
                err_str = str(e_hy)
                quota_terms = ["zerogpu", "quota", "exceeded", "429", "rate limit", "hour", "gpu quota"]
                if is_local_gpu_available and any(t in err_str.lower() for t in quota_terms):
                    print(f"[App] Quota ZeroGPU Hugging Face atteint ({err_str}).")
                    print("[App] Bascule de secours automatique vers le moteur local Hunyuan3D-2 GPU RTX 4070 Ti...")
                    update_progress(35, "Quota Cloud Hugging Face atteint : bascule automatique sur votre moteur local Hunyuan3D-2 (GPU RTX 4070 Ti)...")
                    TASKS[task_id]["warning"] = "Quota gratuit quotidien Hugging Face ZeroGPU atteint. Modélisation effectuée avec succès via le moteur local Hunyuan3D-2 sur votre GPU RTX 4070 Ti (Illimité & Gratuit)."
                    mesh = generate_figurine_360_hunyuan_local(
                        image_path=image_path,
                        target_height_mm=target_height_mm,
                        include_base=include_base,
                        base_height_mm=base_height_mm,
                        base_size_mm=base_size_mm,
                        base_shape=base_shape,
                        steps=30,
                        octree_resolution=octree_res,
                        progress_callback=update_progress
                    )
                else:
                    raise e_hy
        else:
            # Mode Hunyuan3D-2 Local (RTX 4070 Ti) par défaut
            octree_res = max(256, min(512, int(resolution)))
            if is_local_gpu_available:
                mesh = generate_figurine_360_hunyuan_local(
                    image_path=image_path,
                    target_height_mm=target_height_mm,
                    include_base=include_base,
                    base_height_mm=base_height_mm,
                    base_size_mm=base_size_mm,
                    base_shape=base_shape,
                    steps=30,
                    octree_resolution=octree_res,
                    progress_callback=update_progress
                )
            else:
                mesh = generate_figurine_360_hunyuan(
                    image_path=image_path,
                    target_height_mm=target_height_mm,
                    include_base=include_base,
                    base_height_mm=base_height_mm,
                    base_size_mm=base_size_mm,
                    base_shape=base_shape,
                    steps=40,
                    octree_resolution=octree_res,
                    hf_token=hf_token,
                    progress_callback=update_progress
                )

        update_progress(85, "Validation des cotes Bambu Lab A1 Mini et export STL / GLB...")
        metrics = validate_and_scale_for_bambu(
            mesh,
            target_height_mm=0,  # Déjà mis à l'échelle dans postprocess_360_figurine
            enforce_watertight=True
        )

        base_export = os.path.join(task_dir, "model_bambu")
        export_all_formats(mesh, base_export)

        update_progress(95, "Rendu de l'aperçu Studio 3D...")
        preview_path = os.path.join(task_dir, "preview_studio.png")
        render_mesh_preview(mesh, preview_path, img_size=512, angle_deg=25.0)

        TASKS[task_id]["progress"] = 100
        TASKS[task_id]["status"] = "completed"
        TASKS[task_id]["step"] = "Figurine 3D 360° générée avec succès !"
        TASKS[task_id]["result"] = {
            "task_id": task_id,
            "mode": mode,
            "warning": TASKS[task_id].get("warning"),
            "include_base": include_base,
            "metrics": metrics,
            "files": {
                "stl": f"/outputs/{task_id}/model_bambu.stl",
                "glb": f"/outputs/{task_id}/model_bambu.glb",
                "obj": f"/outputs/{task_id}/model_bambu.obj",
                "preview": f"/outputs/{task_id}/preview_studio.png"
            }
        }
        print(f"[App] Tâche 360° {task_id} terminée avec succès.")

    except Exception as e:
        import traceback
        traceback.print_exc()
        TASKS[task_id]["status"] = "failed"
        TASKS[task_id]["error"] = str(e)
        TASKS[task_id]["step"] = f"Erreur : {str(e)}"


@app.post("/api/convert")
async def start_conversion(
    request: Request,
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    demo_image_path: Optional[str] = Form(None),
    mode: str = Form("hunyuan_360_local"),
    target_height_mm: float = Form(120.0),
    include_base: str = Form("true"),
    base_height_mm: float = Form(5.0),
    base_size_mm: float = Form(60.0),
    base_shape: str = Form("auto"),
    resolution: int = Form(384),
    hf_token: Optional[str] = Form(None)
):
    """
    Lance une tâche de modélisation de figurine 3D 360° en arrière-plan (Hunyuan3D uniquement, accès protégé).
    """
    token = request.headers.get("X-Site-Auth") or request.cookies.get(AUTH_COOKIE_NAME)
    if token != AUTH_TOKEN_VALUE:
        raise HTTPException(status_code=401, detail="Accès protégé. Veuillez saisir le mot de passe du site.")

    # Moteur Hunyuan exclusivement
    if mode not in ("hunyuan_360_local", "hunyuan_360"):
        mode = "hunyuan_360_local"
    task_id = str(uuid.uuid4())[:8]
    task_upload_dir = os.path.join(UPLOADS_DIR, f"task_{task_id}")
    os.makedirs(task_upload_dir, exist_ok=True)

    local_img_path = None

    if file and file.filename:
        local_img_path = os.path.join(task_upload_dir, file.filename)
        with open(local_img_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    elif demo_image_path:
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
        raise HTTPException(status_code=400, detail="Veuillez sélectionner ou déposer une image.")

    has_base = str(include_base).lower() in ("true", "1", "yes", "on")

    TASKS[task_id] = {
        "id": task_id,
        "status": "processing",
        "progress": 5,
        "step": "Initialisation du pipeline 3D 360°...",
        "created_at": time.time(),
        "result": None,
        "error": None
    }

    background_tasks.add_task(
        background_generation_task,
        task_id=task_id,
        image_path=local_img_path,
        mode=mode,
        target_height_mm=float(target_height_mm),
        include_base=has_base,
        base_height_mm=float(base_height_mm),
        base_size_mm=float(base_size_mm),
        base_shape=str(base_shape),
        resolution=int(resolution),
        hf_token=hf_token
    )

    return {
        "task_id": task_id,
        "status": "processing",
        "message": "Modélisation 3D 360° démarrée."
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
        raise HTTPException(status_code=400, detail="Format non supporté")

    ext = ext_map[file_format.lower()]
    file_path = os.path.join(OUTPUTS_DIR, task_id, f"model_bambu.{ext}")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Fichier non trouvé")

    return FileResponse(
        file_path,
        filename=f"figurine_360_{task_id}.{ext}",
        media_type="application/octet-stream"
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860 if os.environ.get("SPACE_ID") else 8080))
    host = "0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1"
    uvicorn.run("app:app", host=host, port=port, reload=False)
