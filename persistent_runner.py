"""
Persistent Watchdog Runner for Studio 3D Bambu Lab (RTX 4070 Ti)
Ensures Uvicorn and Cloudflare Tunnel run 24/7 with auto-recovery and stability.
Automatically synchronizes the active tunnel URL with GitHub Pages for a permanent URL.
"""

import os
import sys
import time
import re
import json
import base64
import atexit
import subprocess
import urllib.request

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLOUDFLARED_PATH = os.path.join(BASE_DIR, "cloudflared.exe")
TUNNEL_LOG = os.path.join(BASE_DIR, "tunnel.log")
PUBLIC_URL_FILE = os.path.join(BASE_DIR, "PUBLIC_URL.txt")
PYTHON_EXEC = r"E:\ComfyUI\venv\Scripts\python.exe" if os.path.exists(r"E:\ComfyUI\venv\Scripts\python.exe") else sys.executable

GITHUB_REPO = "jeromefoguenne-eng/Modelisation-3D"
PERMANENT_LINK_SHORT = "https://tinyurl.com/studio3d-edufog"
PERMANENT_LINK_PAGES = "https://jeromefoguenne-eng.github.io/Modelisation-3D/"

def get_github_token():
    # 1. Variable d'environnement
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    # 2. Fichier local .github_token
    token_file = os.path.join(BASE_DIR, ".github_token")
    if os.path.exists(token_file):
        try:
            with open(token_file, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            pass
    # 3. Remote git origin
    try:
        out = subprocess.check_output(["git", "config", "--get", "remote.origin.url"], cwd=BASE_DIR, text=True)
        m = re.search(r":([^@]+)@github\.com", out)
        if m:
            cred = m.group(1)
            if cred.startswith("ghp_"):
                return cred
            if ":" in cred:
                return cred.split(":")[-1]
    except Exception:
        pass
    return None

def update_github_live_url(url, status="online"):
    """
    Met à jour instantanément active_tunnel.json sur GitHub via l'API REST.
    Permet au lien permanent de rediriger immédiatement les étudiants vers l'URL active.
    """
    token = get_github_token()
    if not token:
        print("[Watchdog] Avertissement : Aucun token GitHub disponible pour synchroniser le statut.")
        return

    path = "active_tunnel.json"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Studio3D-Watchdog"
    }
    
    sha = None
    try:
        req = urllib.request.Request(f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}", headers=headers)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            sha = data.get("sha")
    except Exception:
        pass

    payload = {
        "message": f"Auto-update tunnel status: {status}",
        "content": base64.b64encode(json.dumps({
            "status": status,
            "url": url,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }, indent=2).encode("utf-8")).decode("utf-8")
    }
    if sha:
        payload["sha"] = sha

    try:
        req_put = urllib.request.Request(
            f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="PUT"
        )
        with urllib.request.urlopen(req_put, timeout=8) as resp:
            if resp.status in (200, 201):
                print(f"[Watchdog] GitHub synchronisé avec succès ({status}) -> {url}")
    except Exception as e:
        print(f"[Watchdog] Attention: échec sync GitHub : {e}")

def is_uvicorn_running():
    try:
        req = urllib.request.Request("http://127.0.0.1:8080/api/health")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False

def extract_tunnel_url():
    if not os.path.exists(TUNNEL_LOG):
        return None
    try:
        with open(TUNNEL_LOG, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        matches = re.findall(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", content)
        if matches:
            return matches[-1]
    except Exception:
        pass
    return None

def main():
    print("=" * 70)
    print("  Studio 3D Figurine (Bambu Lab A1 Mini - RTX 4070 Ti)")
    print("  SERVICE DE REDIRECTION PERMANENTE ACTIF (24/7)")
    print("=" * 70)
    print(f"LIEN PERMANENT POUR VOS ETUDIANTS : {PERMANENT_LINK_SHORT}")
    print(f"   (ou lien direct GitHub Pages   : {PERMANENT_LINK_PAGES})")
    print("=" * 70)

    uvicorn_proc = None
    cloudflared_proc = None
    last_known_url = None

    while True:
        try:
            # 1. Vérifier / Démarrer Uvicorn
            if not is_uvicorn_running():
                print("[Watchdog] Démarrage du serveur FastAPI local...")
                uvicorn_cmd = [PYTHON_EXEC, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8080"]
                uvicorn_proc = subprocess.Popen(
                    uvicorn_cmd,
                    cwd=BASE_DIR,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                time.sleep(3)

            # 2. Vérifier / Démarrer Cloudflared
            cf_alive = cloudflared_proc and cloudflared_proc.poll() is None
            if not cf_alive:
                print("[Watchdog] Démarrage du tunnel Cloudflare sécurisé...")
                if os.path.exists(TUNNEL_LOG):
                    try:
                        os.remove(TUNNEL_LOG)
                    except Exception:
                        pass

                cf_cmd = [
                    CLOUDFLARED_PATH,
                    "tunnel",
                    "--url", "http://127.0.0.1:8080",
                    "--logfile", TUNNEL_LOG
                ]
                cloudflared_proc = subprocess.Popen(
                    cf_cmd,
                    cwd=BASE_DIR,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                time.sleep(6)

            # 3. Extraire et synchroniser l'URL si elle a changé
            url = extract_tunnel_url()
            if url and url != last_known_url:
                last_known_url = url
                with open(PUBLIC_URL_FILE, "w", encoding="utf-8") as f:
                    f.write(url)
                print(f"\n[Watchdog] >>> SERVEUR ACTIF : {url} <<<")
                print(f"[Watchdog] Mise à jour du lien permanent...")
                update_github_live_url(url, status="online")
                print(f"[Watchdog] ✅ Vos étudiants peuvent se connecter sur : {PERMANENT_LINK_SHORT}\n")

            time.sleep(10)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[Watchdog] Erreur surveillant : {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
