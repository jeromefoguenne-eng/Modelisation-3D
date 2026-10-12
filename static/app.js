/**
 * Application Client Bambu Lab A1 Mini - Studio 3D
 * Gestion Three.js, communication asynchrone FastAPI, zéro fallback dégradé.
 */

let scene, camera, renderer, controls;
let currentMesh = null;
let currentGLBUrl = null;
let currentTaskPollInterval = null;

let selectedFile = null;
let selectedDemoPath = null;
let selectedMode = "bas_relief";

// Initialisation au chargement de la page
window.addEventListener("DOMContentLoaded", () => {
  initThreeJS();
  initEventListeners();
  loadSystemHealth();
  loadDemoGallery();
});

// 1. Initialisation Three.js
function initThreeJS() {
  const container = document.getElementById("canvas-container");
  const width = container.clientWidth;
  const height = container.clientHeight;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0f1115);

  camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
  camera.position.set(0, 40, 220);

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;

  container.appendChild(renderer.domElement);

  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.maxDistance = 500;
  controls.minDistance = 20;

  // Éclairage Studio PBR
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
  scene.add(ambientLight);

  // Lumière principale (Key light)
  const keyLight = new THREE.DirectionalLight(0xffffff, 1.2);
  keyLight.position.set(100, 150, 100);
  keyLight.castShadow = true;
  scene.add(keyLight);

  // Lumière de débouchage (Fill light bleue)
  const fillLight = new THREE.DirectionalLight(0x90b0e0, 0.7);
  fillLight.position.set(-100, 80, -80);
  scene.add(fillLight);

  // Lumière de contour (Rim light)
  const rimLight = new THREE.DirectionalLight(0xffffff, 0.8);
  rimLight.position.set(0, -100, -100);
  scene.add(rimLight);

  // Plateau d'impression Bambu Lab A1 Mini (Grille 180x180 mm)
  const gridHelper = new THREE.GridHelper(180, 18, 0x00ae42, 0x2e3545);
  gridHelper.position.y = 0;
  scene.add(gridHelper);

  // Boucle de rendu
  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  // Redimensionnement réactif
  window.addEventListener("resize", () => {
    const w = container.clientWidth;
    const h = container.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  });
}

// 2. Événements et UI
function initEventListeners() {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const widthSlider = document.getElementById("width-slider");
  const widthVal = document.getElementById("width-val");
  const reliefSlider = document.getElementById("relief-slider");
  const reliefVal = document.getElementById("relief-val");
  const curveSlider = document.getElementById("curve-slider");
  const curveVal = document.getElementById("curve-val");

  // Sliders Voie 2
  widthSlider.addEventListener("input", (e) => {
    widthVal.textContent = `${e.target.value} mm`;
  });
  reliefSlider.addEventListener("input", (e) => {
    reliefVal.textContent = `${e.target.value} mm`;
  });
  curveSlider.addEventListener("input", (e) => {
    const val = parseInt(e.target.value);
    curveVal.textContent = val === 0 ? "0° (Plat)" : `${val}° (Courbé)`;
  });

  // Drag & drop
  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  // Sélection de mode
  document.querySelectorAll(".mode-card").forEach((card) => {
    card.addEventListener("click", () => {
      document.querySelectorAll(".mode-card").forEach((c) => c.classList.remove("active"));
      card.classList.add("active");
      selectedMode = card.getAttribute("data-mode");
    });
  });

  // Matériaux Three.js
  document.querySelectorAll(".tool-btn[data-mat]").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tool-btn[data-mat]").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      applyMaterial(btn.getAttribute("data-mat"));
    });
  });

  // Recentrer vue
  document.getElementById("btn-reset-cam").addEventListener("click", () => {
    if (currentMesh) {
      const box = new THREE.Box3().setFromObject(currentMesh);
      const center = box.getCenter(new THREE.Vector3());
      controls.target.copy(center);
      camera.position.set(center.x, center.y + 40, center.z + 180);
    } else {
      camera.position.set(0, 40, 220);
      controls.target.set(0, 0, 0);
    }
    controls.update();
  });

  // Bouton de lancement
  btnGenerate.addEventListener("click", startConversionProcess);
}

// 3. Gestion du fichier sélectionné
function handleFileSelected(file) {
  selectedFile = file;
  selectedDemoPath = null;
  document.querySelectorAll(".demo-thumb").forEach((t) => t.classList.remove("selected"));

  const reader = new FileReader();
  reader.onload = (e) => {
    const previewImg = document.getElementById("source-preview");
    previewImg.src = e.target.result;
    document.getElementById("preview-box").style.display = "block";
    document.getElementById("btn-generate").disabled = false;
  };
  reader.readAsDataURL(file);
}

// 4. Galerie Démo
async function loadDemoGallery() {
  try {
    const res = await fetch("/api/demo-images");
    const data = await res.json();
    const strip = document.getElementById("demo-strip");
    strip.innerHTML = "";

    if (data.demos && data.demos.length > 0) {
      data.demos.forEach((d) => {
        const img = document.createElement("img");
        img.src = d.url;
        img.className = "demo-thumb";
        img.title = d.name;
        img.addEventListener("click", () => {
          document.querySelectorAll(".demo-thumb").forEach((t) => t.classList.remove("selected"));
          img.classList.add("selected");
          selectedDemoPath = d.local_path;
          selectedFile = null;

          const previewImg = document.getElementById("source-preview");
          previewImg.src = d.url;
          document.getElementById("preview-box").style.display = "block";
          document.getElementById("btn-generate").disabled = false;
        });
        strip.appendChild(img);
      });
    }
  } catch (err) {
    console.error("Erreur chargement démos :", err);
  }
}

// 5. Statut Système
async function loadSystemHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (data.gpu && data.gpu.cuda_available) {
      document.getElementById("gpu-dot").style.backgroundColor = "var(--accent-bambu)";
      document.getElementById("gpu-text").textContent = `${data.gpu.device_name} (${data.gpu.vram_gb} Go VRAM)`;
    } else {
      document.getElementById("gpu-dot").style.backgroundColor = "var(--accent-amber)";
      document.getElementById("gpu-text").textContent = "Mode CPU (Pas de GPU CUDA détecté)";
    }
  } catch (err) {
    console.error("Health check error :", err);
  }
}

// 6. Lancement de la conversion
async function startConversionProcess() {
  const btn = document.getElementById("btn-generate");
  btn.disabled = true;

  const overlay = document.getElementById("progress-overlay");
  const progressBar = document.getElementById("progress-bar");
  const progressStep = document.getElementById("progress-step");
  const progressTitle = document.getElementById("progress-title");

  overlay.style.display = "flex";
  progressBar.style.width = "5%";
  progressStep.textContent = "Préparation de la requête...";
  progressTitle.textContent = "Calcul de la géométrie 3D...";

  const formData = new FormData();
  if (selectedFile) {
    formData.append("file", selectedFile);
  } else if (selectedDemoPath) {
    formData.append("demo_image_path", selectedDemoPath);
  }

  formData.append("mode", selectedMode);
  formData.append("target_width_mm", document.getElementById("width-slider").value);
  formData.append("relief_depth_mm", document.getElementById("relief-slider").value);
  formData.append("curvature_deg", document.getElementById("curve-slider").value);

  try {
    const res = await fetch("/api/convert", {
      method: "POST",
      body: formData
    });

    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.detail || "Erreur lors de l'initialisation de la tâche");
    }

    const data = await res.json();
    const taskId = data.task_id;

    // Polling régulier de la tâche
    pollTaskStatus(taskId);

  } catch (err) {
    overlay.style.display = "none";
    btn.disabled = false;
    alert(`Erreur : ${err.message}`);
  }
}

// 7. Polling de la tâche
function pollTaskStatus(taskId) {
  if (currentTaskPollInterval) {
    clearInterval(currentTaskPollInterval);
  }

  const progressBar = document.getElementById("progress-bar");
  const progressStep = document.getElementById("progress-step");
  const overlay = document.getElementById("progress-overlay");
  const btn = document.getElementById("btn-generate");

  currentTaskPollInterval = setInterval(async () => {
    try {
      const res = await fetch(`/api/task/${taskId}`);
      if (!res.ok) return;

      const task = await res.json();
      progressBar.style.width = `${task.progress}%`;
      progressStep.textContent = task.step || "Traitement en cours...";

      if (task.status === "completed") {
        clearInterval(currentTaskPollInterval);
        overlay.style.display = "none";
        btn.disabled = false;

        // Afficher les résultats
        handleTaskSuccess(task.result);

      } else if (task.status === "failed") {
        clearInterval(currentTaskPollInterval);
        overlay.style.display = "none";
        btn.disabled = false;
        alert(`Échec de la modélisation 3D : ${task.error || "Erreur inconnue"}`);
      }
    } catch (err) {
      console.error("Erreur de suivi :", err);
    }
  }, 1200);
}

// 8. Succès de la modélisation et affichage 3D
function handleTaskSuccess(result) {
  document.getElementById("empty-viewport").style.display = "none";

  // Mise à jour des métriques Bambu Lab A1 Mini
  const m = result.metrics;
  document.getElementById("val-dims").textContent = `${m.dimensions_mm.x} × ${m.dimensions_mm.y} × ${m.dimensions_mm.z} mm`;
  document.getElementById("val-volume").textContent = `${m.volume_cm3} cm³`;
  document.getElementById("val-weight").textContent = `${m.weight_pla_g} g (PLA)`;
  document.getElementById("val-time").textContent = `${m.print_time_estimate}`;

  const watertightBadge = document.getElementById("val-watertight");
  if (m.is_watertight) {
    watertightBadge.textContent = "✔ Watertight (Solide Étanche)";
    watertightBadge.className = "badge-watertight";
  } else {
    watertightBadge.textContent = "⚠ Géométrie non étanche";
    watertightBadge.className = "badge-watertight warning";
  }

  // Configuration des téléchargements
  document.getElementById("btn-dl-stl").href = result.files.stl;
  document.getElementById("btn-dl-glb").href = result.files.glb;
  document.getElementById("metrics-panel").style.display = "flex";

  // Chargement du GLB dans Three.js
  loadGLBModel(result.files.glb);
}

// 9. Chargement Three.js du fichier GLB
function loadGLBModel(glbUrl) {
  currentGLBUrl = glbUrl;
  const loader = new THREE.GLTFLoader();

  loader.load(
    glbUrl,
    (gltf) => {
      if (currentMesh) {
        scene.remove(currentMesh);
      }

      currentMesh = gltf.scene;

      // Centrage géométrique
      const box = new THREE.Box3().setFromObject(currentMesh);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());

      // Poser la base au sol (Z = 0)
      currentMesh.position.x = -center.x;
      currentMesh.position.y = -box.min.y;
      currentMesh.position.z = -center.z;

      // Matériau par défaut PLA blanc Bambu
      applyMaterial("white");

      scene.add(currentMesh);

      // Cadrage de caméra
      const maxDim = Math.max(size.x, size.y, size.z);
      camera.position.set(0, size.y * 0.6, maxDim * 1.8);
      controls.target.set(0, size.y * 0.45, 0);
      controls.update();

      console.log("[ThreeJS] Modèle chargé avec succès :", glbUrl);
    },
    (xhr) => {
      console.log(`[ThreeJS] Chargement GLB : ${(xhr.loaded / xhr.total * 100)}%`);
    },
    (err) => {
      console.error("[ThreeJS] Erreur chargement GLB :", err);
      alert("Erreur de chargement du modèle 3D dans le visualiseur.");
    }
  );
}

// 10. Changement de matériaux PBR
function applyMaterial(type) {
  if (!currentMesh) return;

  currentMesh.traverse((child) => {
    if (child.isMesh) {
      let mat;
      if (type === "white") {
        mat = new THREE.MeshStandardMaterial({
          color: 0xf5f5f7,
          roughness: 0.35,
          metalness: 0.05
        });
      } else if (type === "bambu-green") {
        mat = new THREE.MeshStandardMaterial({
          color: 0x00ae42,
          roughness: 0.3,
          metalness: 0.1
        });
      } else if (type === "bronze") {
        mat = new THREE.MeshStandardMaterial({
          color: 0x9c7a4b,
          roughness: 0.45,
          metalness: 0.8
        });
      } else if (type === "marble") {
        mat = new THREE.MeshStandardMaterial({
          color: 0xeae8e1,
          roughness: 0.2,
          metalness: 0.0
        });
      } else if (type === "wireframe") {
        mat = new THREE.MeshBasicMaterial({
          color: 0x00ae42,
          wireframe: true
        });
      }
      child.material = mat;
      child.castShadow = true;
      child.receiveShadow = true;
    }
  });
}
