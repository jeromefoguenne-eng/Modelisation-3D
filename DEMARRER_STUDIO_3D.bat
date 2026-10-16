@echo off
chcp 65001 > nul
title Studio 3D Figurine - Démarrage Automatique
echo ======================================================================
echo    DÉMARRAGE DU SERVICE STUDIO 3D (RTX 4070 Ti)
echo ======================================================================
echo Lancement du service silencieux en arrière-plan...

start "" wscript.exe "C:\Users\TrendingPC\.gemini\antigravity\scratch\Modelisation-3D\Studio3D_AutoStart.vbs"

timeout /t 3 > nul
echo.
echo Le service tourne en arrière-plan (aucun terminal encombrant).
echo Vous pouvez fermer cette fenêtre.
echo.
pause
