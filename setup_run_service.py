import os

content = """@echo off
cd /d "C:\\Users\\TrendingPC\\.gemini\\antigravity\\scratch\\Modelisation-3D"
start "Studio3D_Watchdog" /MIN "E:\\ComfyUI\\venv\\Scripts\\python.exe" persistent_runner.py
exit
"""

target = r"C:\Users\TrendingPC\.gemini\antigravity\scratch\Modelisation-3D\run_service.bat"
with open(target, "wb") as f:
    f.write(content.replace("\n", "\r\n").encode("ascii"))

print("Created run_service.bat:", os.path.exists(target))
