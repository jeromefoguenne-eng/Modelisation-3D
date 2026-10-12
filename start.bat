@echo off
title Studio 3D Bambu Lab A1 Mini - RTX 4070 Ti
echo ============================================================
echo   Studio 3D Haute Precision - Bambu Lab A1 Mini (RTX 4070 Ti)
echo ============================================================
echo.

set PYTHON_EXEC=E:\ComfyUI\venv\Scripts\python.exe
if not exist "%PYTHON_EXEC%" (
    set PYTHON_EXEC=python
)

echo Utilisation de l'interpreteur : %PYTHON_EXEC%
echo Demarrage du serveur FastAPI sur http://127.0.0.1:8080 ...
echo.

"%PYTHON_EXEC%" -m uvicorn app:app --host 127.0.0.1 --port 8080

pause
