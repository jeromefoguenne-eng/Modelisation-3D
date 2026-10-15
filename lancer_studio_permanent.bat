@echo off
title Studio 3D Figurine 360 - Serveur Permanent
echo ============================================================
echo   Studio 3D Figurine 360 - Mode Permanent & Haute Stabilite
echo ============================================================
echo.
set PYTHON_EXEC=E:\ComfyUI\venv\Scripts\python.exe
if not exist "%PYTHON_EXEC%" (
    set PYTHON_EXEC=python
)

echo Lancement du service surveillant (Uvicorn + Cloudflare)...
"%PYTHON_EXEC%" persistent_runner.py
pause
