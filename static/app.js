/**
 * Application Client Studio 3D Figurine 360° (Bambu Lab A1 Mini)
 * Gestion Three.js 360°, Avec ou Sans Socle, GPU RTX 4070 Ti & Hunyuan3D-2.
 */

let scene, camera, renderer, controls;
let currentMesh = null;
let currentTaskPollInterval = null;

let selectedFile = null;
let selectedDemoPath = null;
let selectedMode = "hunyuan_360_local";
let includeBase = true;
let selectedBaseShape = "auto";

let isShowingPreloadedDemo = true;
let currentPlaColor = "#f0f2f5";
const AUTH_TOKEN_KEY = "studio3d_session";
const SITE_PASSWORD_FALLBACK = "fmttnhech";

window.addEventListener("DOMContentLoaded", () => {
  checkAuthentication();
  initThreeJS();
  initEventListeners();
  loadSystemHealth();
  loadDemoGallery();
  loadPreloadedSotaShowcase(true);
});

function loadPreloadedSotaShowcase(withBase) {
  const metrics = {
    dimensions_mm: { x: 62.4, y: 120.0, z: 58.2 },
    volume_cm3: 38.5,
    weight_pla_g: 18.1,
    print_time_estimate: "52 min",
    face_count: 534920
  };

  handleTaskSuccess({
    include_base: withBase,
    metrics: metrics,
    files: {
      stl: `/outputs/version_clown/model_bambu.stl`,
      glb: `/outputs/version_clown/model_bambu.glb`
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

  camera = new THREE.PerspectiveCamera(45, width / height, 1.0, 1000);
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
  controls.maxDistance = 600;
  controls.minDistance = 20;
  controls.autoRotate = false;
  controls.autoRotateSpeed = 2.0;

  // Éclairage Studio PBR 360° Haute Visibilité
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.95);
  scene.add(ambientLight);

  const keyLight = new THREE.DirectionalLight(0xfff8ee, 1.8);
  keyLight.position.set(120, 180, 120);
  keyLight.castShadow = true;
  keyLight.shadow.mapSize.width = 2048;
  keyLight.shadow.mapSize.height = 2048;
  keyLight.shadow.camera.near = 10;
  keyLight.shadow.camera.far = 450;
  keyLight.shadow.camera.left = -150;
  keyLight.shadow.camera.right = 150;
  keyLight.shadow.camera.top = 150;
  keyLight.shadow.camera.bottom = -150;
  keyLight.shadow.bias = -0.0002;
  keyLight.shadow.normalBias = 0.04;
  scene.add(keyLight);

  const fillLight = new THREE.DirectionalLight(0xa5c5f5, 1.25);
  fillLight.position.set(-120, 100, -100);
  scene.add(fillLight);

  const rimLight = new THREE.DirectionalLight(0xffffff, 1.3);
  rimLight.position.set(0, 140, -150);
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
  const baseSizeGroup = document.getElementById("base-size-group");
  const baseShapeGroup = document.getElementById("base-shape-group");

  const baseSlider = document.getElementById("base-slider");
  const baseVal = document.getElementById("base-val");
  const baseSizeSlider = document.getElementById("base-size-slider");
  const baseSizeVal = document.getElementById("base-size-val");
  const heightSlider = document.getElementById("height-slider");
  const heightVal = document.getElementById("height-val");
  const resSlider = document.getElementById("res-slider");
  const resVal = document.getElementById("res-val");

  // Authentification par mot de passe
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

  // Boutons Avec / Sans Socle
  btnWithBase.addEventListener("click", () => {
    includeBase = true;
    btnWithBase.classList.add("active");
    btnWithoutBase.classList.remove("active");
    baseHeightGroup.style.display = "block";
    if (baseSizeGroup) baseSizeGroup.style.display = "block";
    if (baseShapeGroup) baseShapeGroup.style.display = "block";
    if (isShowingPreloadedDemo) {
      loadPreloadedSotaShowcase(true);
    }
  });

  btnWithoutBase.addEventListener("click", () => {
    includeBase = false;
    btnWithoutBase.classList.add("active");
    btnWithBase.classList.remove("active");
    baseHeightGroup.style.display = "none";
    if (baseSizeGroup) baseSizeGroup.style.display = "none";
    if (baseShapeGroup) baseShapeGroup.style.display = "none";
    if (isShowingPreloadedDemo) {
      loadPreloadedSotaShowcase(false);
    }
  });

  // Slider Taille / Diamètre du socle
  if (baseSizeSlider && baseSizeVal) {
    baseSizeSlider.addEventListener("input", (e) => {
      baseSizeVal.textContent = `${e.target.value} mm`;
    });
  }

  // Forme du socle (Auto / Rond / Ovale)
  document.querySelectorAll(".base-shape-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".base-shape-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      selectedBaseShape = btn.getAttribute("data-shape") || "auto";
    });
  });

  // Sliders
  baseSlider.addEventListener("input", (e) => {
    baseVal.textContent = `${e.target.value} mm`;
  });

  // Préréglages rapides de hauteur (20mm, 28mm, 50mm, 100mm)
  document.querySelectorAll(".height-preset-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".height-preset-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const h = btn.getAttribute("data-height");
      heightSlider.value = h;
      heightVal.textContent = `${h} mm`;

      // Si miniature ou micro pion (<= 30mm), adapter automatiquement la taille du socle pour des proportions parfaites
      const hInt = parseInt(h);
      if (hInt <= 30 && baseSizeSlider) {
        let recommendedBase = 25;
        let recommendedThickness = 2.5;
        if (hInt <= 12) {
          recommendedBase = 10;
          recommendedThickness = 1.5;
        } else if (hInt <= 22) {
          recommendedBase = 18;
          recommendedThickness = 2.0;
        }
        if (parseInt(baseSizeSlider.value) > recommendedBase + 5) {
          baseSizeSlider.value = recommendedBase;
          if (baseSizeVal) baseSizeVal.textContent = `${recommendedBase} mm`;
        }
        if (baseSlider && parseFloat(baseSlider.value) > recommendedThickness + 0.5) {
          baseSlider.value = recommendedThickness;
          if (baseVal) baseVal.textContent = `${recommendedThickness} mm`;
        }
      }
    });
  });

  heightSlider.addEventListener("input", (e) => {
    const val = e.target.value;
    heightVal.textContent = `${val} mm`;
    // Mettre à jour l'état actif des boutons préréglés
    document.querySelectorAll(".height-preset-btn").forEach((b) => {
      if (b.getAttribute("data-height") === val) {
        b.classList.add("active");
      } else {
        b.classList.remove("active");
      }
    });
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

  // Bouton PLA et sélecteur de couleur interactif
  const btnPla = document.getElementById("btn-pla");
  const plaPicker = document.getElementById("pla-color-picker");
  const plaPreview = document.getElementById("pla-color-preview");
  const plaLabel = document.getElementById("pla-label");

  if (btnPla && plaPicker) {
    btnPla.addEventListener("click", (e) => {
      // Déclenche l'ouverture de la palette de couleurs au clic
      plaPicker.click();
    });

    plaPicker.addEventListener("input", (e) => {
      currentPlaColor = e.target.value;
      if (plaPreview) plaPreview.style.backgroundColor = currentPlaColor;
      if (plaLabel) {
        plaLabel.textContent = (currentPlaColor.toLowerCase() === "#f0f2f5" || currentPlaColor.toLowerCase() === "#ffffff")
          ? "PLA Blanc"
          : `PLA (${currentPlaColor.toUpperCase()})`;
      }
      document.querySelectorAll(".tool-btn[data-mat]").forEach((b) => b.classList.remove("active"));
      btnPla.classList.add("active");
      applyMaterial("white");
    });
  }

  // Auto-rotation 360° (Désactivée par défaut, s'active au clic)
  const btnAutoRotate = document.getElementById("btn-autorotate");
  btnAutoRotate.addEventListener("click", () => {
    controls.autoRotate = !controls.autoRotate;
    btnAutoRotate.textContent = controls.autoRotate ? "🔄 Rotation 360°: ON" : "🔄 Rotation 360°: OFF";
    btnAutoRotate.classList.toggle("active", controls.autoRotate);
  });

  // Recentrer vue
  document.getElementById("btn-reset-cam").addEventListener("click", recenterCamera);

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

        // Pré-sélectionner la première image par défaut pour que le bouton soit actif d'office
        if (idx === 0) {
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

  formData.append("mode", "hunyuan_360_local");
  formData.append("target_height_mm", document.getElementById("height-slider").value);
  formData.append("include_base", includeBase ? "true" : "false");
  formData.append("base_height_mm", document.getElementById("base-slider").value);
  formData.append("base_size_mm", document.getElementById("base-size-slider") ? document.getElementById("base-size-slider").value : "50");
  formData.append("base_shape", selectedBaseShape);
  formData.append("resolution", document.getElementById("res-slider").value);

  try {
    const sessionToken = localStorage.getItem(AUTH_TOKEN_KEY) || "";
    const res = await fetch("/api/convert", {
      method: "POST",
      headers: {
        "X-Site-Auth": sessionToken
      },
      body: formData
    });

    if (res.status === 401) {
      overlay.style.display = "none";
      btn.disabled = false;
      localStorage.removeItem(AUTH_TOKEN_KEY);
      const authOverlay = document.getElementById("auth-overlay");
      if (authOverlay) {
        authOverlay.style.display = "flex";
        authOverlay.style.opacity = "1";
        document.getElementById("auth-error-msg").textContent = "Session expirée. Veuillez entrer le mot de passe.";
      }
      return;
    }

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

        if (task.result && task.result.warning) {
          const banner = document.getElementById("gpu-quota-banner");
          const bannerText = document.getElementById("gpu-quota-banner-text");
          if (banner && bannerText) {
            bannerText.textContent = "⚡ " + task.result.warning;
            banner.style.display = "flex";
          }
        }

      } else if (task.status === "failed") {
        clearInterval(currentTaskPollInterval);
        overlay.style.display = "none";
        btn.disabled = false;
        const err = task.error || "Erreur inconnue";
        if (err.toLowerCase().includes("zerogpu") || err.toLowerCase().includes("quota") || err.toLowerCase().includes("429")) {
          alert("Quota gratuit quotidien Hugging Face ZeroGPU atteint (60s max / 24h sur cette adresse IP).\n\n💡 Solution immédiate :\n1. Cliquez sur le mode 'TripoSR Local (RTX 4070 Ti)' dans le menu de gauche : il est 100% hors-ligne, gratuit et illimité !\n2. Ou renseignez un token Hugging Face personnel dans le champ 'Clé API Hugging Face'.");
        } else {
          alert(`Échec de la modélisation : ${err}`);
        }
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

      // Sauvegarder le matériau d'origine sur chaque sous-maillage et fiabiliser les normales
      currentMesh.traverse((child) => {
        if (child.isMesh) {
          // Recalcul et normalisation des normales si absentes ou corrompues
          if (!child.geometry.attributes.normal) {
            child.geometry.computeVertexNormals();
          }
          child.geometry.normalizeNormals();

          // Projeter les ombres mais désactiver l'auto-ombrage sur le maillage pour stopper le scintillement (shadow acne)
          child.castShadow = true;
          child.receiveShadow = false;

          child.userData.originalMaterial = child.material;
          child.userData.hasTexture = !!(child.material && child.material.map);
          child.userData.hasVertexColors = !!(child.geometry && child.geometry.attributes && child.geometry.attributes.color);
        }
      });

      // Appliquer le matériau actif dans la barre d'outils
      const activeMatBtn = document.querySelector(".tool-btn[data-mat].active");
      const matType = activeMatBtn ? activeMatBtn.getAttribute("data-mat") : "vertex";
      applyMaterial(matType);

      scene.add(currentMesh);

      // Cadrage automatique centré et lumineux pour statuette 360°
      recenterCamera();
    },
    undefined,
    (err) => {
      console.error("[ThreeJS] Erreur chargement GLB :", err);
      alert("Erreur de chargement du modèle 3D dans le visualiseur.");
    }
  );
}

// Recentrage parfait de la figurine dans l'espace 3D (visibilité intégrale sans scroll, zoom fluide)
function recenterCamera() {
  if (!currentMesh) {
    camera.position.set(0, 70, 200);
    controls.target.set(0, 45, 0);
    controls.update();
    return;
  }
  currentMesh.updateMatrixWorld(true);
  const worldBox = new THREE.Box3().setFromObject(currentMesh);
  const worldCenter = worldBox.getCenter(new THREE.Vector3());
  const worldSize = worldBox.getSize(new THREE.Vector3());

  const container = document.getElementById("canvas-container");
  const width = (container && container.clientWidth) ? container.clientWidth : 800;
  const height = (container && container.clientHeight) ? container.clientHeight : 600;
  const aspect = width / height;

  const vFovRad = (camera.fov * Math.PI) / 180;
  const hFovRad = 2 * Math.atan(Math.tan(vFovRad / 2) * aspect);

  // Distance calculée pour faire entrer toute la figurine (hauteur et largeur) avec marge de respiration
  const distY = (worldSize.y / 2) / Math.tan(vFovRad / 2) * 1.25;
  const distX = (Math.max(worldSize.x, worldSize.z) / 2) / Math.tan(hFovRad / 2) * 1.25;
  // S'adapte parfaitement aux micro pions de 10mm comme aux grandes statuettes de 175mm
  const dist = Math.max(distY, distX, 14);

  // Mire légèrement abaissée pour dégager le socle au-dessus de la barre métrique
  const targetY = worldCenter.y - (worldSize.y * 0.06);
  controls.target.set(0, targetY, 0);

  // Caméra positionnée face au modèle
  camera.position.set(0, targetY + (worldSize.y * 0.12), dist);
  camera.near = 0.1;
  camera.far = Math.max(3000, dist * 10);
  camera.updateProjectionMatrix();

  // Contrôles de zoom fluides (molette et tactile, zoom macro sur micro pions)
  controls.enableZoom = true;
  controls.minDistance = 1.2;
  controls.maxDistance = dist * 5;
  controls.update();
}

// Vérification de l'authentification (mot de passe "fmttnhech")
function checkAuthentication() {
  const overlay = document.getElementById("auth-overlay");
  if (!overlay) return;

  const sessionToken = localStorage.getItem(AUTH_TOKEN_KEY);
  const localPwd = localStorage.getItem("studio3d_password");

  if (sessionToken === "auth_token_fmttnhech_verified" || localPwd === SITE_PASSWORD_FALLBACK) {
    overlay.style.display = "none";
  } else {
    overlay.style.display = "flex";
    setTimeout(() => {
      const inp = document.getElementById("site-password-input");
      if (inp) inp.focus();
    }, 150);
  }
}

// Soumission du mot de passe
async function handleAuthSubmit() {
  const pwdInput = document.getElementById("site-password-input");
  const errMsg = document.getElementById("auth-error-msg");
  const btnSubmit = document.getElementById("btn-submit-auth");
  const overlay = document.getElementById("auth-overlay");

  if (!pwdInput) return;
  const pwd = pwdInput.value.trim();
  errMsg.textContent = "";

  if (!pwd) {
    errMsg.textContent = "Veuillez entrer le mot de passe.";
    return;
  }

  try {
    btnSubmit.disabled = true;
    btnSubmit.textContent = "Vérification...";

    const res = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password: pwd })
    });

    const data = await res.json();
    if (res.ok && data.success) {
      localStorage.setItem(AUTH_TOKEN_KEY, data.token);
      localStorage.setItem("studio3d_password", pwd);
      document.cookie = `studio3d_session=${data.token}; path=/; max-age=2592000; SameSite=Lax`;
      overlay.style.transition = "opacity 0.3s ease";
      overlay.style.opacity = "0";
      setTimeout(() => {
        overlay.style.display = "none";
        overlay.style.opacity = "1";
      }, 300);
    } else {
      errMsg.textContent = data.detail || "Mot de passe incorrect.";
      pwdInput.focus();
      pwdInput.select();
    }
  } catch (err) {
    // Repli de secours local si le mot de passe correspond
    if (pwd === SITE_PASSWORD_FALLBACK) {
      localStorage.setItem(AUTH_TOKEN_KEY, "auth_token_fmttnhech_verified");
      localStorage.setItem("studio3d_password", pwd);
      overlay.style.display = "none";
    } else {
      errMsg.textContent = "Mot de passe incorrect.";
    }
  } finally {
    btnSubmit.disabled = false;
    btnSubmit.textContent = "Déverrouiller l'accès";
  }
}

// 10. Changement de matériaux PBR
function applyMaterial(type) {
  if (!currentMesh) return;

  currentMesh.traverse((child) => {
    if (child.isMesh) {
      if (type === "vertex") {
        if (child.userData.hasTexture && child.userData.originalMaterial) {
          child.material = child.userData.originalMaterial;
          child.material.roughness = 0.38;
          child.material.metalness = 0.05;
          child.material.side = THREE.DoubleSide;
          child.material.needsUpdate = true;
        } else if (child.userData.hasVertexColors) {
          child.material = new THREE.MeshStandardMaterial({
            vertexColors: true,
            roughness: 0.45,
            metalness: 0.05,
            side: THREE.DoubleSide
          });
        }
      } else if (type === "white" || (type === "vertex" && !child.userData.hasVertexColors && !child.userData.hasTexture)) {
        child.material = new THREE.MeshStandardMaterial({
          color: new THREE.Color(currentPlaColor || "#f0f2f5"),
          roughness: 0.35,
          metalness: 0.08,
          side: THREE.DoubleSide
        });
      } else if (type === "bambu-green") {
        child.material = new THREE.MeshStandardMaterial({
          color: 0x00ae42,
          roughness: 0.3,
          metalness: 0.1,
          side: THREE.DoubleSide
        });
      } else if (type === "bronze") {
        child.material = new THREE.MeshStandardMaterial({
          color: 0xad8347,
          roughness: 0.38,
          metalness: 0.78,
          side: THREE.DoubleSide
        });
      } else if (type === "marble") {
        child.material = new THREE.MeshStandardMaterial({
          color: 0xeae8e1,
          roughness: 0.22,
          metalness: 0.02,
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
}
