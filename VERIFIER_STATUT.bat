@echo off
chcp 65001 > nul
title Statut Studio 3D
echo ======================================================================
echo    VÉRIFICATION DE L'ÉTAT DE STUDIO 3D & LIENS ÉTUDIANTS
echo ======================================================================
echo.
python -c "import urllib.request, json, os; base = r'C:\Users\TrendingPC\.gemini\antigravity\scratch\Modelisation-3D'; f_active = os.path.join(base, 'active_tunnel.json'); data = json.load(open(f_active, encoding='utf-8')) if os.path.exists(f_active) else {'status': 'inconnu', 'url': 'N/A'}; print('Statut serveur   :', data.get('status')); print('URL active        :', data.get('url')); print('Derniere maj      :', data.get('updated_at')); print('\nLien court etudiants : https://tinyurl.com/studio3d-edufog'); print('Portail GitHub Pages : https://jeromefoguenne-eng.github.io/Modelisation-3D/')"
echo.
echo ======================================================================
pause
