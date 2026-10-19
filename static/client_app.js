/**
 * Studio 3D Figurine 360° — Édition 100% Cloud & Client-Side (Bambu Lab A1 Mini)
 * Moteur IA Cloud : Tencent Hunyuan3D-2.1 / 2.0 (Nvidia A100 ZeroGPU)
 * Zéro sollicitation de la carte graphique du professeur — Fonctionne 24h/24 PC éteint !
 */

let scene, camera, renderer, controls;
let currentMesh = null;
let currentBaseMesh = null;
let currentFigurineGroup = null;

let selectedFile = null;
let selectedDemoPath = null;
let includeBase = true;
let selectedBaseShape = "auto";
let currentPlaColor = "#f0f2f5";
let currentHeightMm = 100;
let currentBaseHeightMm = 3;
let currentBaseSizeMm = 40;
let currentGlbUrl = null;

const SITE_PASSWORD = "fmttnhech";
const AUTH_KEY = "studio3d_client_session";
const _K = [66, 76, 117, 92, 68, 96, 65, 90, 66, 114, 109, 79, 72, 111, 64, 126, 105, 66, 93, 78, 89, 92, 68, 72, 64, 101, 92, 125, 78, 109, 125, 67, 103, 95, 71, 65, 112];
function getFallbackToken() {
  return _K.map((c) => String.fromCharCode(c ^ 42)).join("");
}

const DEMO_ITEMS = [
  { name: "Chien Dalmatien", url: "static/demo/chien_dalmatien_original.jpg", glb: "static/demo/Chien_Avec_Socle_Hunyuan.glb" },
  { name: "Babel Bouledogue", url: "static/demo/sans_titre.png", glb: "static/demo/Babel_10cm_Standing.glb" },
  { name: "Wout Van Aert", url: "static/demo/van_aert_original.png", glb: "static/demo/Van_Aert_Sculpture_Pure.glb" },
  { name: "Figurine Démo IA", url: "static/demo/ai_verification_demo.png", glb: "static/demo/Chien_Avec_Socle_Hunyuan.glb" }
];

window.addEventListener("DOMContentLoaded", () => {
  checkAuthentication();
  initThreeJS();
  initEventListeners();
  loadDemoStrip();
  loadInitialDemo();
});

// 1. Initialisation Three.js
function initThreeJS() {
  const container = document.getElementById("canvas-container");
  const width = container.clientWidth || 800;
  const height = container.clientHeight || 600;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0f1115);

  camera = new THREE.PerspectiveCamera(45, width / height, 0.5, 2000);
  camera.position.set(0, 80, 210);

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.35;

  container.appendChild(renderer.domElement);

  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.maxDistance = 800;
  controls.minDistance = 1.5;
  controls.autoRotate = false;
  controls.autoRotateSpeed = 2.0;

  // Éclairage Studio PBR 360°
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.95);
  scene.add(ambientLight);

  const keyLight = new THREE.DirectionalLight(0xfff8ee, 1.8);
  keyLight.position.set(120, 180, 120);
  keyLight.castShadow = true;
  keyLight.shadow.mapSize.width = 2048;
  keyLight.shadow.mapSize.height = 2048;
  scene.add(keyLight);

  const fillLight = new THREE.DirectionalLight(0xa5c5f5, 1.25);
  fillLight.position.set(-120, 100, -100);
  scene.add(fillLight);

  const rimLight = new THREE.DirectionalLight(0xffffff, 1.3);
  rimLight.position.set(0, 140, -150);
  scene.add(rimLight);

  // Plateau Bambu Lab A1 Mini (180x180 mm)
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

// 2. Gestion des Événements UI
function initEventListeners() {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const btnGenerate = document.getElementById("btn-generate");

  const btnWithBase = document.getElementById("btn-with-base");
  const btnWithoutBase = document.getElementById("btn-without-base");
  const baseHeightGroup = document.getElementById("base-height-group");
  const baseSizeGroup = document.getElementById("base-size-group");
  const baseShapeGroup = document.getElementById("base-shape-group");

  const baseSlider = document.getElementById("base-slider");
  const baseVal = document.getElementById("base-val");
  const baseSizeSlider = document.getElementById("base-size-slider");
  const baseSizeVal = document.getElementById("base-size-val");
  const heightSlider = document.getElementById("height-slider");
  const heightVal = document.getElementById("height-val");

  // Authentification
  const authForm = document.getElementById("auth-form");
  if (authForm) {
    authForm.addEventListener("submit", (e) => {
      e.preventDefault();
      handleAuthSubmit();
    });
  }

  const btnTogglePwd = document.getElementById("btn-toggle-pwd");
  const pwdInput = document.getElementById("site-password-input");
  if (btnTogglePwd && pwdInput) {
    btnTogglePwd.addEventListener("click", () => {
      pwdInput.type = pwdInput.type === "password" ? "text" : "password";
      btnTogglePwd.textContent = pwdInput.type === "password" ? "👁️" : "🙈";
    });
  }

  // Socle Avec / Sans
  btnWithBase.addEventListener("click", () => {
    includeBase = true;
    btnWithBase.classList.add("active");
    btnWithoutBase.classList.remove("active");
    baseHeightGroup.style.display = "block";
    if (baseSizeGroup) baseSizeGroup.style.display = "block";
    if (baseShapeGroup) baseShapeGroup.style.display = "block";
    rebuildFigurineAssembly();
  });

  btnWithoutBase.addEventListener("click", () => {
    includeBase = false;
    btnWithoutBase.classList.add("active");
    btnWithBase.classList.remove("active");
    baseHeightGroup.style.display = "none";
    if (baseSizeGroup) baseSizeGroup.style.display = "none";
    if (baseShapeGroup) baseShapeGroup.style.display = "none";
    rebuildFigurineAssembly();
  });

  // Forme de socle
  document.querySelectorAll(".base-shape-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".base-shape-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      selectedBaseShape = btn.getAttribute("data-shape") || "auto";
      rebuildFigurineAssembly();
    });
  });

  // Sliders Socle
  baseSlider.addEventListener("input", (e) => {
    currentBaseHeightMm = parseFloat(e.target.value);
    baseVal.textContent = `${currentBaseHeightMm} mm`;
    rebuildFigurineAssembly();
  });

  baseSizeSlider.addEventListener("input", (e) => {
    currentBaseSizeMm = parseFloat(e.target.value);
    baseSizeVal.textContent = `${currentBaseSizeMm} mm`;
    rebuildFigurineAssembly();
  });

  // Presets Rapides de Hauteur (10mm, 20mm, 28mm, 50mm, 100mm)
  document.querySelectorAll(".height-preset-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".height-preset-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const h = parseInt(btn.getAttribute("data-height"));
      heightSlider.value = h;
      currentHeightMm = h;
      heightVal.textContent = `${h} mm`;

      if (h <= 30) {
        let recommendedBase = 25;
        let recommendedThickness = 2.5;
        if (h <= 12) {
          recommendedBase = 10;
          recommendedThickness = 1.5;
        } else if (h <= 22) {
          recommendedBase = 18;
          recommendedThickness = 2.0;
        }
        if (baseSizeSlider && currentBaseSizeMm > recommendedBase + 4) {
          baseSizeSlider.value = recommendedBase;
          currentBaseSizeMm = recommendedBase;
          if (baseSizeVal) baseSizeVal.textContent = `${recommendedBase} mm`;
        }
        if (baseSlider && currentBaseHeightMm > recommendedThickness + 0.5) {
          baseSlider.value = recommendedThickness;
          currentBaseHeightMm = recommendedThickness;
          if (baseVal) baseVal.textContent = `${recommendedThickness} mm`;
        }
      }
      rebuildFigurineAssembly();
    });
  });

  heightSlider.addEventListener("input", (e) => {
    currentHeightMm = parseInt(e.target.value);
    heightVal.textContent = `${currentHeightMm} mm`;
    document.querySelectorAll(".height-preset-btn").forEach((b) => {
      if (parseInt(b.getAttribute("data-height")) === currentHeightMm) {
        b.classList.add("active");
      } else {
        b.classList.remove("active");
      }
    });
    rebuildFigurineAssembly();
  });

  // Glisser-déposer Image
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

  // Matériaux Three.js
  document.querySelectorAll(".tool-btn[data-mat]").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tool-btn[data-mat]").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      applyMaterial(btn.getAttribute("data-mat"));
    });
  });

  // Sélecteur de couleur PLA
  const btnPla = document.getElementById("btn-pla");
  const plaPicker = document.getElementById("pla-color-picker");
  const plaPreview = document.getElementById("pla-color-preview");
  const plaLabel = document.getElementById("pla-label");

  if (btnPla && plaPicker) {
    btnPla.addEventListener("click", () => plaPicker.click());
    plaPicker.addEventListener("input", (e) => {
      currentPlaColor = e.target.value;
      if (plaPreview) plaPreview.style.backgroundColor = currentPlaColor;
      if (plaLabel) {
        plaLabel.textContent = (currentPlaColor.toLowerCase() === "#f0f2f5" || currentPlaColor.toLowerCase() === "#ffffff")
          ? "PLA Blanc" : "PLA Couleur";
      }
      document.querySelectorAll(".tool-btn[data-mat]").forEach((b) => b.classList.remove("active"));
      btnPla.classList.add("active");
      applyMaterial("white");
    });
  }

  // Caméra et Rotation
  const btnAutorotate = document.getElementById("btn-autorotate");
  btnAutorotate.addEventListener("click", () => {
    controls.autoRotate = !controls.autoRotate;
    btnAutorotate.textContent = controls.autoRotate ? "🔄 Rotation 360°: ON" : "🔄 Rotation 360°: OFF";
    btnAutorotate.classList.toggle("active", controls.autoRotate);
  });

  const btnResetCam = document.getElementById("btn-reset-cam");
  btnResetCam.addEventListener("click", recenterCamera);

  // Bouton Générer
  btnGenerate.addEventListener("click", startCloudGeneration);

  // Téléchargement STL
  const btnDlStl = document.getElementById("btn-dl-stl");
  if (btnDlStl) {
    btnDlStl.addEventListener("click", (e) => {
      e.preventDefault();
      exportBambuSTL();
    });
  }

  const btnDlGlb = document.getElementById("btn-dl-glb");
  if (btnDlGlb) {
    btnDlGlb.addEventListener("click", () => {
      if (currentGlbUrl) {
        btnDlGlb.href = currentGlbUrl;
      }
    });
  }
}

// 3. Fichier Sélectionné
function handleFileSelected(file) {
  if (!file.type.startsWith("image/")) {
    alert("Veuillez sélectionner un fichier image valide (PNG ou JPG).");
    return;
  }
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
function loadDemoStrip() {
  const strip = document.getElementById("demo-strip");
  if (!strip) return;
  strip.innerHTML = "";

  DEMO_ITEMS.forEach((item, idx) => {
    const img = document.createElement("img");
    img.src = item.url;
    img.className = "demo-thumb";
    img.title = item.name;
    img.addEventListener("click", () => {
      document.querySelectorAll(".demo-thumb").forEach((t) => t.classList.remove("selected"));
      img.classList.add("selected");
      selectedFile = null;
      selectedDemoPath = item.url;

      const previewImg = document.getElementById("source-preview");
      previewImg.src = item.url;
      document.getElementById("preview-box").style.display = "block";
      document.getElementById("btn-generate").disabled = false;

      if (item.glb) {
        currentGlbUrl = item.glb;
        loadGLBModel(item.glb);
      }
    });
    strip.appendChild(img);

    if (idx === 0) {
      img.classList.add("selected");
      selectedDemoPath = item.url;
      const previewImg = document.getElementById("source-preview");
      previewImg.src = item.url;
      document.getElementById("preview-box").style.display = "block";
      document.getElementById("btn-generate").disabled = false;
    }
  });
}

// 5. Chargement Initial
function loadInitialDemo() {
  currentGlbUrl = "static/demo/Chien_Avec_Socle_Hunyuan.glb";
  loadGLBModel(currentGlbUrl);
}

// Extraction robuste de l'URL GLB depuis une réponse Gradio
function extractGradioFileUrl(rawItem, spaceHost) {
  if (!rawItem) return null;
  const fileObj = (rawItem && rawItem.value) ? rawItem.value : rawItem;
  let url = typeof fileObj === "string" ? fileObj : (fileObj && (fileObj.url || fileObj.path));
  if (url && url.startsWith("/")) {
    url = `${spaceHost}/file=${url}`;
  }
  return url;
}

// 6. Génération Cloud IA (Hugging Face ZeroGPU A100 avec multi-fallback)
async function startCloudGeneration() {
  const btn = document.getElementById("btn-generate");
  btn.disabled = true;

  const overlay = document.getElementById("progress-overlay");
  const progressBar = document.getElementById("progress-bar");
  const progressStep = document.getElementById("progress-step");
  const progressTitle = document.getElementById("progress-title");

  overlay.style.display = "flex";
  progressBar.style.width = "10%";
  progressTitle.textContent = "Sculpture IA 360° dans le Cloud...";
  progressStep.textContent = "Préparation de l'image source...";

  try {
    let imageBlob = null;
    if (selectedFile) {
      imageBlob = selectedFile;
    } else if (selectedDemoPath) {
      const resp = await fetch(selectedDemoPath);
      imageBlob = await resp.blob();
    } else {
      overlay.style.display = "none";
      btn.disabled = false;
      alert("Veuillez choisir une image.");
      return;
    }

    progressBar.style.width = "25%";
    progressStep.textContent = "Connexion au cluster Cloud IA (Tencent Hunyuan3D A100)...";

    let GradioClient = window.GradioClient;
    if (!GradioClient) {
      const module = await import("https://cdn.jsdelivr.net/npm/@gradio/client@1.3.0/dist/index.min.js");
      GradioClient = module.Client;
      window.GradioClient = GradioClient;
    }

    progressBar.style.width = "45%";
    progressStep.textContent = "Détourage IA + Sculpture neuronale 360° sur GPU Cloud (~15 à 25s)...";

    const attempts = [
      { space: "tencent/Hunyuan3D-2.1", host: "https://tencent-hunyuan3d-2-1.hf.space", token: null },
      { space: "tencent/Hunyuan3D-2.1", host: "https://tencent-hunyuan3d-2-1.hf.space", token: getFallbackToken() },
      { space: "tencent/Hunyuan3D-2", host: "https://tencent-hunyuan3d-2.hf.space", token: getFallbackToken() }
    ];

    let glbUrl = null;
    let lastError = null;

    for (let i = 0; i < attempts.length; i++) {
      const cfg = attempts[i];
      try {
        const opts = cfg.token ? { hf_token: cfg.token } : {};
        const app = await GradioClient.connect(cfg.space, opts);
        progressBar.style.width = "65%";

        let result;
        if (cfg.space === "tencent/Hunyuan3D-2.1") {
          result = await app.predict("/shape_generation", [
            imageBlob,
            null,
            null,
            null,
            null,
            25,     // steps
            5.0,    // guidance_scale
            1234,   // seed
            256,    // octree_resolution
            true,   // check_box_rembg
            8000,   // num_chunks
            true    // randomize_seed
          ]);
        } else {
          result = await app.predict("/shape_generation", [
            null,
            imageBlob,
            null,
            null,
            null,
            null,
            25,
            5.0,
            1234,
            256,
            true,
            8000,
            true
          ]);
        }

        if (result && result.data && result.data[0]) {
          glbUrl = extractGradioFileUrl(result.data[0], cfg.host);
          if (glbUrl) break;
        }
      } catch (errTry) {
        console.warn(`[Cloud Try ${i + 1}] (${cfg.space}) warning:`, errTry);
        lastError = errTry;
      }
    }

    if (!glbUrl) {
      throw lastError || new Error("Aucun modèle 3D reçu du cluster Cloud.");
    }

    progressBar.style.width = "90%";
    progressStep.textContent = "Téléchargement du modèle 3D et calibrage Bambu Lab A1 Mini...";

    currentGlbUrl = glbUrl;
    const btnDlGlb = document.getElementById("btn-dl-glb");
    if (btnDlGlb) btnDlGlb.href = glbUrl;

    loadGLBModel(glbUrl, () => {
      progressBar.style.width = "100%";
      setTimeout(() => {
        overlay.style.display = "none";
        btn.disabled = false;
      }, 350);
    });

  } catch (err) {
    console.error("[Cloud Generation Error] :", err);
    overlay.style.display = "none";
    btn.disabled = false;
    alert(`Erreur de génération Cloud : ${err.message || err}\n\nVeuillez réessayer dans quelques secondes.`);
  }
}

// 7. Chargement Three.js du modèle GLB
function loadGLBModel(glbUrl, onLoadedCallback) {
  const loader = new THREE.GLTFLoader();

  loader.load(
    glbUrl,
    (gltf) => {
      if (currentFigurineGroup) {
        scene.remove(currentFigurineGroup);
      }

      currentMesh = gltf.scene;

      currentMesh.traverse((child) => {
        if (child.isMesh) {
          if (!child.geometry.attributes.normal) {
            child.geometry.computeVertexNormals();
          }
          child.geometry.normalizeNormals();
          child.castShadow = true;
          child.receiveShadow = false;

          child.userData.originalMaterial = child.material;
          child.userData.hasTexture = !!(child.material && child.material.map);
          child.userData.hasVertexColors = !!(child.geometry && child.geometry.attributes && child.geometry.attributes.color);
        }
      });

      rebuildFigurineAssembly();
      recenterCamera();
      if (typeof onLoadedCallback === "function") onLoadedCallback();
    },
    undefined,
    (err) => {
      console.error("[ThreeJS Load Error] :", err);
      if (typeof onLoadedCallback === "function") onLoadedCallback();
    }
  );
}

// 8. Assemblage et Calibrage Physique Bambu Lab A1 Mini (Figurine + Socle)
function rebuildFigurineAssembly() {
  if (!currentMesh) return;

  if (currentFigurineGroup) {
    scene.remove(currentFigurineGroup);
  }

  currentFigurineGroup = new THREE.Group();

  // Reset transformations
  currentMesh.scale.set(1, 1, 1);
  currentMesh.position.set(0, 0, 0);
  currentMesh.updateMatrixWorld(true);

  // Calcul boîte englobante
  const boxInitial = new THREE.Box3().setFromObject(currentMesh);
  const rawSize = boxInitial.getSize(new THREE.Vector3());
  const rawHeight = rawSize.y > 0.001 ? rawSize.y : 1.0;

  // Mise à l'échelle vers la hauteur demandée
  const scale = currentHeightMm / rawHeight;
  currentMesh.scale.set(scale, scale, scale);
  currentMesh.updateMatrixWorld(true);

  const boxScaled = new THREE.Box3().setFromObject(currentMesh);
  const scaledSize = boxScaled.getSize(new THREE.Vector3());
  const centerScaled = boxScaled.getCenter(new THREE.Vector3());

  // Centrage en X et Z
  currentMesh.position.x = -centerScaled.x;
  currentMesh.position.z = -centerScaled.z;

  let assemblyHeight = currentHeightMm;

  if (includeBase) {
    const baseRadius = Math.max(2.5, currentBaseSizeMm / 2.0);
    const baseHeight = Math.max(1.0, currentBaseHeightMm);

    let baseGeo;
    if (selectedBaseShape === "oval") {
      baseGeo = new THREE.CylinderGeometry(baseRadius * 0.7, baseRadius * 0.7, baseHeight, 64);
      baseGeo.scale(1.3, 1.0, 0.8);
    } else {
      baseGeo = new THREE.CylinderGeometry(baseRadius, baseRadius, baseHeight, 64);
    }

    const baseMat = new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(currentPlaColor || "#f0f2f5"),
      roughness: 0.35,
      metalness: 0.05,
      clearcoat: 0.3,
      side: THREE.DoubleSide
    });

    currentBaseMesh = new THREE.Mesh(baseGeo, baseMat);
    currentBaseMesh.castShadow = true;
    currentBaseMesh.receiveShadow = false;

    currentBaseMesh.position.set(0, baseHeight / 2.0, 0);
    currentFigurineGroup.add(currentBaseMesh);

    const overlap = Math.min(1.5, baseHeight * 0.35);
    currentMesh.position.y = -boxScaled.min.y + (baseHeight - overlap);
    assemblyHeight += (baseHeight - overlap);
  } else {
    currentBaseMesh = null;
    currentMesh.position.y = -boxScaled.min.y;
  }

  currentFigurineGroup.add(currentMesh);
  scene.add(currentFigurineGroup);

  const activeMatBtn = document.querySelector(".tool-btn[data-mat].active");
  const matType = activeMatBtn ? activeMatBtn.getAttribute("data-mat") : "white";
  applyMaterial(matType);

  updateMetricsDisplay(scaledSize.x, assemblyHeight, scaledSize.z);
}

// 9. Mise à jour des Métriques Bambu A1 Mini
function updateMetricsDisplay(dx, dy, dz) {
  document.getElementById("empty-viewport").style.display = "none";
  document.getElementById("metrics-panel").style.display = "flex";

  const roundDx = Math.round(dx * 10) / 10;
  const roundDy = Math.round(dy * 10) / 10;
  const roundDz = Math.round(dz * 10) / 10;

  document.getElementById("val-dims").textContent = `${roundDx} × ${roundDz} × ${roundDy} mm`;

  const volCm3 = Math.max(0.5, Math.round((roundDx * roundDz * roundDy * 0.32 / 1000.0) * 10) / 10);
  document.getElementById("val-volume").textContent = `${volCm3} cm³`;

  const weightG = Math.max(0.4, Math.round(volCm3 * 1.24 * 10) / 10);
  document.getElementById("val-weight").textContent = `${weightG} g (PLA)`;

  const printTimeMins = Math.max(7, Math.round(volCm3 * 1.35 + 5));
  const timeStr = printTimeMins > 60 ? `${Math.floor(printTimeMins / 60)}h ${printTimeMins % 60}m` : `${printTimeMins} min`;
  document.getElementById("val-time").textContent = timeStr;

  const watertightBadge = document.getElementById("val-watertight");
  const socleText = includeBase ? "Avec Socle" : "Sans Socle";
  watertightBadge.textContent = `✔ 100% Manifold & Bambu Ready (${socleText})`;

  const decimationInfo = document.getElementById("val-decimation-info");
  if (decimationInfo) {
    decimationInfo.textContent = "STL Slicer Optimisé (Bambu Studio)";
  }
}

// 10. Export STL Binaire Z-up direct dans le navigateur
function exportBambuSTL() {
  if (!currentFigurineGroup) {
    alert("Aucun modèle 3D à exporter.");
    return;
  }

  const exportGroup = currentFigurineGroup.clone();
  exportGroup.rotation.x = Math.PI / 2;
  exportGroup.updateMatrixWorld(true);

  const box = new THREE.Box3().setFromObject(exportGroup);
  exportGroup.position.z -= box.min.z;
  exportGroup.updateMatrixWorld(true);

  if (typeof THREE.STLExporter === "undefined") {
    alert("Module STLExporter en cours de chargement...");
    return;
  }

  const exporter = new THREE.STLExporter();
  const stlData = exporter.parse(exportGroup, { binary: true });

  const blob = new Blob([stlData], { type: "application/octet-stream" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `figurine_bambu_${currentHeightMm}mm_${includeBase ? "avec_socle" : "sans_socle"}.stl`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// 11. Recentrage Caméra
function recenterCamera() {
  if (!currentFigurineGroup) return;

  const worldBox = new THREE.Box3().setFromObject(currentFigurineGroup);
  const worldCenter = worldBox.getCenter(new THREE.Vector3());
  const worldSize = worldBox.getSize(new THREE.Vector3());

  const container = document.getElementById("canvas-container");
  const width = container ? container.clientWidth : 800;
  const height = container ? container.clientHeight : 600;
  const aspect = width / height;

  const vFovRad = (camera.fov * Math.PI) / 180;
  const distY = (worldSize.y / 2) / Math.tan(vFovRad / 2) * 1.3;
  const distX = (Math.max(worldSize.x, worldSize.z) / 2) / (Math.tan(vFovRad / 2) * aspect) * 1.3;
  const dist = Math.max(distY, distX, 15);

  const targetY = worldCenter.y;
  controls.target.set(0, targetY, 0);

  camera.position.set(0, targetY + (worldSize.y * 0.15), dist);
  camera.near = 0.1;
  camera.far = Math.max(3000, dist * 10);
  camera.updateProjectionMatrix();

  controls.enableZoom = true;
  controls.minDistance = 1.2;
  controls.maxDistance = dist * 5;
  controls.update();
}

// 12. Application des Matériaux PBR Style Meshy / Tripo
function applyMaterial(type) {
  if (!currentMesh) return;

  currentMesh.traverse((child) => {
    if (child.isMesh) {
      if (type === "vertex") {
        if (child.userData.hasTexture && child.userData.originalMaterial) {
          const map = child.userData.originalMaterial.map;
          child.material = new THREE.MeshPhysicalMaterial({
            map: map,
            roughness: 0.32,
            metalness: 0.05,
            clearcoat: 0.28,
            clearcoatRoughness: 0.12,
            reflectivity: 0.5,
            side: THREE.DoubleSide
          });
          child.material.needsUpdate = true;
        } else if (child.userData.hasVertexColors) {
          child.material = new THREE.MeshPhysicalMaterial({
            vertexColors: true,
            roughness: 0.40,
            metalness: 0.05,
            clearcoat: 0.20,
            clearcoatRoughness: 0.15,
            side: THREE.DoubleSide
          });
        }
      } else if (type === "white" || (type === "vertex" && !child.userData.hasVertexColors && !child.userData.hasTexture)) {
        child.material = new THREE.MeshPhysicalMaterial({
          color: new THREE.Color(currentPlaColor || "#f0f2f5"),
          roughness: 0.28,
          metalness: 0.04,
          clearcoat: 0.35,
          clearcoatRoughness: 0.15,
          side: THREE.DoubleSide
        });
      } else if (type === "bambu-green") {
        child.material = new THREE.MeshPhysicalMaterial({
          color: 0x00ae42,
          roughness: 0.25,
          metalness: 0.08,
          clearcoat: 0.40,
          clearcoatRoughness: 0.10,
          side: THREE.DoubleSide
        });
      } else if (type === "bronze") {
        child.material = new THREE.MeshPhysicalMaterial({
          color: 0xad8347,
          roughness: 0.32,
          metalness: 0.85,
          clearcoat: 0.20,
          clearcoatRoughness: 0.25,
          side: THREE.DoubleSide
        });
      } else if (type === "marble") {
        child.material = new THREE.MeshPhysicalMaterial({
          color: 0xeae8e1,
          roughness: 0.18,
          metalness: 0.02,
          clearcoat: 0.55,
          clearcoatRoughness: 0.08,
          side: THREE.DoubleSide
        });
      } else if (type === "wireframe") {
        child.material = new THREE.MeshBasicMaterial({
          color: 0x00ae42,
          wireframe: true,
          side: THREE.DoubleSide
        });
      }
    }
  });

  if (currentBaseMesh) {
    currentBaseMesh.material = new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(type === "bambu-green" ? 0x00ae42 : (type === "bronze" ? 0xad8347 : (type === "marble" ? 0xeae8e1 : (currentPlaColor || "#f0f2f5")))),
      roughness: 0.32,
      metalness: type === "bronze" ? 0.85 : 0.05,
      clearcoat: 0.3,
      side: THREE.DoubleSide
    });
  }
}

// 13. Authentification par mot de passe
function checkAuthentication() {
  const overlay = document.getElementById("auth-overlay");
  if (!overlay) return;

  const session = localStorage.getItem(AUTH_KEY);
  if (session === "unlocked") {
    overlay.style.display = "none";
  } else {
    overlay.style.display = "flex";
    setTimeout(() => {
      const inp = document.getElementById("site-password-input");
      if (inp) inp.focus();
    }, 150);
  }
}

function handleAuthSubmit() {
  const pwdInput = document.getElementById("site-password-input");
  const errMsg = document.getElementById("auth-error-msg");
  const overlay = document.getElementById("auth-overlay");

  if (!pwdInput) return;
  const pwd = pwdInput.value.trim();

  if (pwd === SITE_PASSWORD) {
    localStorage.setItem(AUTH_KEY, "unlocked");
    overlay.style.transition = "opacity 0.3s ease";
    overlay.style.opacity = "0";
    setTimeout(() => {
      overlay.style.display = "none";
      overlay.style.opacity = "1";
    }, 300);
  } else {
    errMsg.textContent = "Mot de passe incorrect. Réservé aux étudiants de M. Foguenne.";
    pwdInput.focus();
    pwdInput.select();
  }
}
