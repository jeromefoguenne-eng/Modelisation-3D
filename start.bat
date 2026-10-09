@echo off
title Modelisation-3D - Bambu Lab A1 Mini
echo ============================================================
echo   Agent Antigravity - PNG to 3D Print (Bambu Lab A1 Mini)
echo ============================================================
echo Demarrage du serveur local FastAPI...
echo.

python -m uvicorn app:app --host 127.0.0.1 --port 8080 --reload
pause
