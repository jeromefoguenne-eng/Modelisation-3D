/**
 * Application Client Studio 3D Figurine 360° (Bambu Lab A1 Mini)
 * Gestion Three.js 360°, Avec ou Sans Socle, GPU RTX 4070 Ti & Hunyuan3D-2.
 */

let scene, camera, renderer, controls;
let currentMesh = null;
let currentTaskPollInterval = null;

let selectedFile = null;
let selectedDemoPath = null;
let selectedMode = "hunyuan_360";
let includeBase = true;

let isShowingPreloadedDemo = true;

window.addEventListener("DOMContentLoaded", () => {
  initThreeJS();
  initEventListeners();
  loadSystemHealth();
  loadDemoGallery();
  loadPreloadedSotaShowcase(true);
});

function loadPreloadedSotaShowcase(withBase) {
  const folder = withBase ? "vanaert_2.7m_with_base" : "vanaert_2.7m_no_base";
  const metrics = withBase
    ? { dimensions_mm: { x: 77.0, y: 125.0, z: 118.5 }, volume_cm3: 70.30, weight_pla_g: 33.1, print_time_estimate: "1h 22m", face_count: 2687654 }
    : { dimensions_mm: { x: 52.8, y: 120.0, z: 106.5 }, volume_cm3: 34.49, weight_pla_g: 16.2, print_time_estimate: "54 min", face_count: 2687270 };

  handleTaskSuccess({
    include_base: withBase,
    metrics: metrics,
    files: {
      stl: `/outputs/${folder}/model_bambu.stl`,
      glb: `/outputs/${folder}/model_bambu.glb`
    }
  });
}

// 1. Initialisation Three.js
function initThreeJS() {
  const container = document.getElementById("canvas-container");
  const width = container.clientWidth || 800;
  const height = container.clientHeight || 600;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0f1115);

  camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 2000);
  camera.position.set(0, 80, 210);

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.15;

  container.appendChild(renderer.domElement);

  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.maxDistance = 600;
  controls.minDistance = 20;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 2.0;

  // Éclairage Studio PBR 360°
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
  scene.add(ambientLight);

  const keyLight = new THREE.DirectionalLight(0xfff6ea, 1.4);
  keyLight.position.set(120, 180, 120);
  keyLight.castShadow = true;
  scene.add(keyLight);

  const fillLight = new THREE.DirectionalLight(0x90b4e8, 0.85);
  fillLight.position.set(-120, 100, -100);
  scene.add(fillLight);

  const rimLight = new THREE.DirectionalLight(0xffffff, 0.9);
  rimLight.position.set(0, 120, -150);
  scene.add(rimLight);

  // Plateau d'impression Bambu Lab A1 Mini (Grille 180x180 mm)
  const gridHelper = new THREE.GridHelper(180, 18, 0x00ae42, 0x2e3545);
  gridHelper.position.y = 0;
  scene.add(gridHelper);

  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

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
  const btnGenerate = document.getElementById("btn-generate");

  const btnWithBase = document.getElementById("btn-with-base");
  const btnWithoutBase = document.getElementById("btn-without-base");
  const baseHeightGroup = document.getElementById("base-height-group");

  const baseSlider = document.getElementById("base-slider");
  const baseVal = document.getElementById("base-val");
  const heightSlider = document.getElementById("height-slider");
  const heightVal = document.getElementById("height-val");
  const resSlider = document.getElementById("res-slider");
  const resVal = document.getElementById("res-val");

  // Boutons Avec / Sans Socle
  btnWithBase.addEventListener("click", () => {
    includeBase = true;
    btnWithBase.classList.add("active");
    btnWithoutBase.classList.remove("active");
    baseHeightGroup.style.display = "flex";
    if (isShowingPreloadedDemo) {
      loadPreloadedSotaShowcase(true);
    }
  });

  btnWithoutBase.addEventListener("click", () => {
    includeBase = false;
    btnWithoutBase.classList.add("active");
    btnWithBase.classList.remove("active");
    baseHeightGroup.style.display = "none";
    if (isShowingPreloadedDemo) {
      loadPreloadedSotaShowcase(false);
    }
  });

  // Sliders
  baseSlider.addEventListener("input", (e) => {
    baseVal.textContent = `${e.target.value} mm`;
  });

  heightSlider.addEventListener("input", (e) => {
    heightVal.textContent = `${e.target.value} mm`;
  });

  resSlider.addEventListener("input", (e) => {
    const r = parseInt(e.target.value);
    const label = r <= 256 ? "Standard" : (r <= 384 ? "Ultra-HD 500k faces" : "Extrême");
    resVal.textContent = `${r}³ (${label})`;
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

  // Sélection du moteur 360°
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

  // Auto-rotation 360°
  const btnAutoRotate = document.getElementById("btn-autorotate");
  btnAutoRotate.addEventListener("click", () => {
    controls.autoRotate = !controls.autoRotate;
    btnAutoRotate.textContent = controls.autoRotate ? "🔄 Rotation 360°: ON" : "⏸ Rotation 360°: OFF";
    btnAutoRotate.classList.toggle("active", controls.autoRotate);
  });

  // Recentrer vue
  document.getElementById("btn-reset-cam").addEventListener("click", () => {
    if (currentMesh) {
      const box = new THREE.Box3().setFromObject(currentMesh);
      const size = box.getSize(new THREE.Vector3());
      controls.target.set(0, size.y * 0.48, 0);
      camera.position.set(0, size.y * 0.65, Math.max(size.x, size.y, size.z) * 1.75);
    } else {
      camera.position.set(0, 80, 210);
      controls.target.set(0, 40, 0);
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

// 4. Galerie Démo (sélectionne automatiquement la première démo pour permettre un clic immédiat)
async function loadDemoGallery() {
  try {
    const res = await fetch("/api/demo-images");
    const data = await res.json();
    const strip = document.getElementById("demo-strip");
    strip.innerHTML = "";

    if (data.demos && data.demos.length > 0) {
      data.demos.forEach((d, idx) => {
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

        // Pré-sélectionner Van Aert ou la première image par défaut pour que le bouton soit actif d'office
        if (idx === 0 || d.name.toLowerCase().includes("van aert")) {
          img.click();
        }
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
      document.getElementById("gpu-text").textContent = "Mode CPU";
    }
  } catch (err) {
    console.error("Health check error :", err);
  }
}

// 6. Lancement de la conversion 360°
async function startConversionProcess() {
  isShowingPreloadedDemo = false;
  const btn = document.getElementById("btn-generate");
  btn.disabled = true;

  const overlay = document.getElementById("progress-overlay");
  const progressBar = document.getElementById("progress-bar");
  const progressStep = document.getElementById("progress-step");
  const progressTitle = document.getElementById("progress-title");

  overlay.style.display = "flex";
  progressBar.style.width = "10%";
  progressStep.textContent = "Envoi de l'image au moteur 360°...";
  progressTitle.textContent = "Sculpture de la Figurine 3D (360°)...";

  const formData = new FormData();
  if (selectedFile) {
    formData.append("file", selectedFile);
  } else if (selectedDemoPath) {
    formData.append("demo_image_path", selectedDemoPath);
  } else {
    overlay.style.display = "none";
    btn.disabled = false;
    alert("Veuillez d'abord choisir une image ou cliquer sur une image de démo.");
    return;
  }

  formData.append("mode", selectedMode);
  formData.append("target_height_mm", document.getElementById("height-slider").value);
  formData.append("include_base", includeBase ? "true" : "false");
  formData.append("base_height_mm", document.getElementById("base-slider").value);
  formData.append("resolution", document.getElementById("res-slider").value);

  try {
    const res = await fetch("/api/convert", {
      method: "POST",
      body: formData
    });

    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.detail || "Erreur lors du lancement de la modélisation 360°");
    }

    const data = await res.json();
    pollTaskStatus(data.task_id);

  } catch (err) {
    overlay.style.display = "none";
    btn.disabled = false;
    alert(`Erreur : ${err.message}`);
  }
}

// 7. Suivi temps réel de la tâche
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
      progressStep.textContent = task.step || "Calcul 360° en cours...";

      if (task.status === "completed") {
        clearInterval(currentTaskPollInterval);
        overlay.style.display = "none";
        btn.disabled = false;
        handleTaskSuccess(task.result);

      } else if (task.status === "failed") {
        clearInterval(currentTaskPollInterval);
        overlay.style.display = "none";
        btn.disabled = false;
        alert(`Échec de la modélisation : ${task.error || "Erreur inconnue"}`);
      }
    } catch (err) {
      console.error("Erreur de suivi :", err);
    }
  }, 1000);
}

// 8. Affichage du résultat 360°
function handleTaskSuccess(result) {
  document.getElementById("empty-viewport").style.display = "none";

  const m = result.metrics;
  document.getElementById("val-dims").textContent = `${m.dimensions_mm.x} × ${m.dimensions_mm.y} × ${m.dimensions_mm.z} mm`;
  document.getElementById("val-volume").textContent = `${m.volume_cm3} cm³`;
  document.getElementById("val-weight").textContent = `${m.weight_pla_g} g (PLA)`;
  document.getElementById("val-time").textContent = `${m.print_time_estimate}`;

  const watertightBadge = document.getElementById("val-watertight");
  const socleText = result.include_base ? "Avec Socle" : "Sans Socle";
  watertightBadge.textContent = `✔ 360° (${socleText})`;

  document.getElementById("btn-dl-stl").href = result.files.stl;
  document.getElementById("btn-dl-glb").href = result.files.glb;
  document.getElementById("metrics-panel").style.display = "flex";

  loadGLBModel(result.files.glb);
}

// 9. Chargement Three.js du modèle 360° GLB
function loadGLBModel(glbUrl) {
  const loader = new THREE.GLTFLoader();

  loader.load(
    `${glbUrl}?t=${Date.now()}`,
    (gltf) => {
      if (currentMesh) {
        scene.remove(currentMesh);
      }

      currentMesh = gltf.scene;

      const box = new THREE.Box3().setFromObject(currentMesh);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());

      // Poser exactement sur la grille (Y = 0)
      currentMesh.position.x = -center.x;
      currentMesh.position.y = -box.min.y;
      currentMesh.position.z = -center.z;

      // Sauvegarder le matériau vertex color d'origine sur chaque sous-maillage
      currentMesh.traverse((child) => {
        if (child.isMesh) {
          child.castShadow = true;
          child.receiveShadow = true;
          child.userData.hasVertexColors = !!(child.geometry && child.geometry.attributes && child.geometry.attributes.color);
        }
      });

      // Appliquer le matériau actif dans la barre d'outils
      const activeMatBtn = document.querySelector(".tool-btn[data-mat].active");
      const matType = activeMatBtn ? activeMatBtn.getAttribute("data-mat") : "vertex";
      applyMaterial(matType);

      scene.add(currentMesh);

      // Cadrage caméra idéal pour statuette 360°
      const maxDim = Math.max(size.x, size.y, size.z);
      camera.position.set(0, size.y * 0.65, maxDim * 1.75);
      controls.target.set(0, size.y * 0.48, 0);
      controls.update();
    },
    undefined,
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
      if (type === "vertex" && child.userData.hasVertexColors) {
        child.material = new THREE.MeshStandardMaterial({
          vertexColors: true,
          roughness: 0.45,
          metalness: 0.05
        });
      } else if (type === "white" || (type === "vertex" && !child.userData.hasVertexColors)) {
        child.material = new THREE.MeshStandardMaterial({
          color: 0xf0f2f5,
          roughness: 0.35,
          metalness: 0.08
        });
      } else if (type === "bambu-green") {
        child.material = new THREE.MeshStandardMaterial({
          color: 0x00ae42,
          roughness: 0.3,
          metalness: 0.1
        });
      } else if (type === "bronze") {
        child.material = new THREE.MeshStandardMaterial({
          color: 0xad8347,
          roughness: 0.38,
          metalness: 0.78
        });
      } else if (type === "marble") {
        child.material = new THREE.MeshStandardMaterial({
          color: 0xeae8e1,
          roughness: 0.22,
          metalness: 0.02
        });
      } else if (type === "wireframe") {
        child.material = new THREE.MeshBasicMaterial({
          color: 0x00ae42,
          wireframe: true
        });
      }
    }
  });
}
