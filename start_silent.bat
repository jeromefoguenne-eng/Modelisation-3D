@echo off
cd /d "C:\Users\TrendingPC\.gemini\antigravity\scratch\Modelisation-3D"
set PYTHONIOENCODING=utf-8
"E:\ComfyUI\venv\Scripts\python.exe" persistent_runner.py > runner_debug.log 2>&1
