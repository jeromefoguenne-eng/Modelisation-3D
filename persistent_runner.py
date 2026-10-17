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
import socket
import subprocess
import urllib.request

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLOUDFLARED_PATH = os.path.join(BASE_DIR, "cloudflared.exe")
TUNNEL_LOG = os.path.join(BASE_DIR, "tunnel.log")
PUBLIC_URL_FILE = os.path.join(BASE_DIR, "PUBLIC_URL.txt")
WATCHDOG_LOG = os.path.join(BASE_DIR, "watchdog.log")
LOCAL_ACTIVE_JSON = os.path.join(BASE_DIR, "active_tunnel.json")
UVICORN_LOG = os.path.join(BASE_DIR, "uvicorn.log")

PYTHON_EXEC = r"E:\ComfyUI\venv\Scripts\python.exe" if os.path.exists(r"E:\ComfyUI\venv\Scripts\python.exe") else sys.executable

GITHUB_REPO = "jeromefoguenne-eng/Modelisation-3D"
PERMANENT_LINK_SHORT = "https://tinyurl.com/studio3d-edufog"
PERMANENT_LINK_PAGES = "https://jeromefoguenne-eng.github.io/Modelisation-3D/"

def log(msg):
    ts = time.strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{ts} {msg}"
    try:
        print(line, flush=True)
    except Exception:
        pass
    try:
        with open(WATCHDOG_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

def get_github_token():
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    token_file = os.path.join(BASE_DIR, ".github_token")
    if os.path.exists(token_file):
        try:
            with open(token_file, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            pass
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
    Met à jour instantanément active_tunnel.json localement ET sur GitHub via l'API REST.
    Permet au lien permanent (GitHub Pages / TinyURL) de rediriger immédiatement les étudiants.
    """
    # 1. Sauvegarde locale
    try:
        with open(LOCAL_ACTIVE_JSON, "w", encoding="utf-8") as f:
            json.dump({
                "status": status,
                "url": url,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2)
    except Exception as e:
        log(f"[Watchdog] Erreur écriture local active_tunnel.json : {e}")

    # 2. Synchronisation GitHub API REST
    token = get_github_token()
    if not token:
        log("[Watchdog] Avertissement : Aucun token GitHub disponible pour synchroniser l'API.")
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
                log(f"[Watchdog] GitHub synchronisé avec succès ({status}) -> {url}")
    except Exception as e:
        log(f"[Watchdog] Attention: échec sync GitHub : {e}")

def is_uvicorn_running():
    try:
        req = urllib.request.Request("http://127.0.0.1:8080/api/health")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False

def is_internet_available():
    try:
        req = urllib.request.Request("https://1.1.1.1", headers={"User-Agent": "Studio3D-Ping/1.0"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            return True
    except Exception:
        return False

def is_tunnel_healthy(url):
    if not url:
        return False
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Studio3D-Watchdog/1.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            return resp.status in (200, 302, 401, 405)
    except urllib.error.HTTPError as e:
        return e.code in (200, 302, 401, 405)
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

def kill_all_cloudflared():
    try:
        subprocess.run(["taskkill", "/F", "/IM", "cloudflared.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

def start_cloudflared():
    kill_all_cloudflared()
    time.sleep(1)
    if os.path.exists(TUNNEL_LOG):
        try:
            os.remove(TUNNEL_LOG)
        except Exception:
            pass

    cf_cmd = [
        CLOUDFLARED_PATH,
        "tunnel",
        "--url", "http://127.0.0.1:8080",
        "--retries", "10",
        "--edge-ip-version", "auto",
        "--logfile", TUNNEL_LOG
    ]
    proc = subprocess.Popen(
        cf_cmd,
        cwd=BASE_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    # Attendre que l'URL apparaisse dans le log
    for _ in range(15):
        time.sleep(1)
        url = extract_tunnel_url()
        if url:
            return proc, url
    return proc, None

def prevent_windows_sleep():
    """
    Empêche Windows d'endormir le CPU et le réseau (Mode Veille suspendu)
    pendant que le serveur Studio 3D est actif (Away Mode).
    """
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001 | 0x00000040)
        log("[Watchdog] Protection anti-mise en veille Windows activée (CPU & réseau maintenus 24/7).")
    except Exception as e:
        log(f"[Watchdog] Note alimentation : {e}")

def main():
    # 1. Mutex Singleton (évite les doublons)
    lock_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        lock_sock.bind(("127.0.0.1", 8099))
        lock_sock.listen(1)
    except Exception:
        log("[Watchdog] Une instance du Watchdog est déjà active sur le système. Arrêt.")
        sys.exit(0)

    prevent_windows_sleep()

    log("=" * 70)
    log("  Studio 3D Figurine (Bambu Lab A1 Mini - RTX 4070 Ti)")
    log("  SERVICE DE SURVEILLANCE & REDIRECTION PERMANENTE 24/7 ACTIF")
    log("=" * 70)
    log(f"Lien permanent étudiants : {PERMANENT_LINK_SHORT}")
    log(f"Portail GitHub Pages    : {PERMANENT_LINK_PAGES}")
    log("=" * 70)

    uvicorn_proc = None
    cloudflared_proc = None
    last_known_url = None
    failed_pings = 0
    last_heartbeat_sync = time.time()

    while True:
        try:
            # 1. Vérifier Uvicorn (FastAPI)
            if not is_uvicorn_running():
                log("[Watchdog] Démarrage du serveur FastAPI local...")
                uvicorn_log_file = open(UVICORN_LOG, "a", encoding="utf-8")
                uvicorn_cmd = [PYTHON_EXEC, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8080"]
                uvicorn_proc = subprocess.Popen(
                    uvicorn_cmd,
                    cwd=BASE_DIR,
                    stdout=uvicorn_log_file,
                    stderr=uvicorn_log_file
                )
                time.sleep(3)

            # 2. Détection de crash immédiat du processus cloudflared
            if cloudflared_proc is not None and cloudflared_proc.poll() is not None:
                log("[Watchdog] Processus Cloudflare interrompu. Relance immédiate...")
                cloudflared_proc, new_url = start_cloudflared()
                if new_url:
                    last_known_url = new_url
                    failed_pings = 0
                    log(f"[Watchdog] Tunnel Cloudflare rétabli : {last_known_url}")
                    with open(PUBLIC_URL_FILE, "w", encoding="utf-8") as f:
                        f.write(last_known_url)
                    update_github_live_url(last_known_url, status="online")

            # 3. Adopter ou vérifier le tunnel Cloudflare
            if not last_known_url:
                # Vérifier si un tunnel préexistant est encore actif et valide
                candidate_url = extract_tunnel_url()
                if candidate_url and is_tunnel_healthy(candidate_url):
                    last_known_url = candidate_url
                    log(f"[Watchdog] Tunnel existant détecté et opérationnel : {last_known_url}")
                    update_github_live_url(last_known_url, status="online")
                else:
                    log("[Watchdog] Initialisation d'un nouveau tunnel Cloudflare...")
                    cloudflared_proc, last_known_url = start_cloudflared()
                    if last_known_url:
                        log(f"[Watchdog] Nouveau tunnel actif : {last_known_url}")
                        with open(PUBLIC_URL_FILE, "w", encoding="utf-8") as f:
                            f.write(last_known_url)
                        update_github_live_url(last_known_url, status="online")
            else:
                # Tester la santé du tunnel actif
                if not is_tunnel_healthy(last_known_url):
                    if not is_internet_available():
                        log("[Watchdog] Connexion Internet interrompue sur le PC. En attente de reconnexion...")
                    else:
                        failed_pings += 1
                        log(f"[Watchdog] Alerte : Tunnel public injoignable ({failed_pings}/2) : {last_known_url}")
                        if failed_pings >= 2:
                            log("[Watchdog] Tunnel Cloudflare expiré ou rompu. Régénération immédiate...")
                            cloudflared_proc, new_url = start_cloudflared()
                            if new_url:
                                last_known_url = new_url
                                failed_pings = 0
                                log(f"[Watchdog] Nouveau tunnel régénéré avec succès : {last_known_url}")
                                with open(PUBLIC_URL_FILE, "w", encoding="utf-8") as f:
                                    f.write(last_known_url)
                                update_github_live_url(last_known_url, status="online")
                else:
                    failed_pings = 0

            # 4. Heartbeat périodique (toutes les 15 minutes) pour maintenir le timestamp frais
            if last_known_url and (time.time() - last_heartbeat_sync > 900):
                log("[Watchdog] Heartbeat 15min : rafraîchissement du statut en ligne...")
                update_github_live_url(last_known_url, status="online")
                last_heartbeat_sync = time.time()

            time.sleep(5)

        except KeyboardInterrupt:
            log("[Watchdog] Arrêt manuel demandé.")
            break
        except Exception as e:
            log(f"[Watchdog] Erreur surveillant : {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
