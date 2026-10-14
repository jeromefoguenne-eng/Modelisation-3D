# Image de base Python 3.10 optimisée pour Hugging Face Spaces (100% Gratuit)
FROM python:3.10-slim

# Dépendances système requises pour OpenCV, OpenGL, Trimesh et Manifold
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Création de l'utilisateur standard Hugging Face Spaces (UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

# Installation des dépendances Python
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copie de l'ensemble du code de l'application
COPY --chown=user:user . .

# Création des dossiers de travail avec permissions d'écriture
RUN mkdir -p uploads outputs && chmod 777 uploads outputs

# Port standard requis par Hugging Face Spaces
EXPOSE 7860

# Lancement du serveur FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
