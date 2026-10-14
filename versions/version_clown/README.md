# Figurine 3D 360° — Version Clown (Release Stable)

## 📌 Présentation de la Version
Cette version fige l'état de l'application avec la résolution complète du problème de détourage et de découpe de la boîte englobante 3D (Bounding Box) sur les figurines hautes/étroites.

## 🎨 Modèle & Géométrie
* **Sujet** : Figurine Clown avec chapeau pointu, pompon sphérique, collerette à volants et boutons.
* **Moteur 3D utilisé** : Tencent Hunyuan3D-2 FP16 (100% Local sur NVIDIA RTX 4070 Ti, 12 Go VRAM).
* **Résolution du maillage** : **534 922 faces** / 267 461 sommets.
* **Intégrité géométrique** : **Watertight** (100% étanche, sans trous ni normales inversées).
* **Dimensions Bambu Lab A1 Mini** : $77.1 \times 54.1 \times 125.0\text{ mm}$ (socle stabilisateur inclus).
* **Temps estimé d'impression** : ~2h07 (PLA Blanc, 0.16mm).

## 📁 Fichiers Inclus
* `model_bambu.stl` : Fichier de fabrication pour Bambu Studio / OrcaSlicer.
* `model_bambu.glb` : Format 3D temps réel avec texturation pour WebGL Three.js.
* `model_bambu.obj` : Maillage universel 3D compatible Blender, Maya, CAD.
* `preview_studio.png` : Rendu de contrôle qualité en studio virtuel.
* `Gemini_Generated_Image_3me5nd3me5nd3me5.jpg` : Photo originale source.
* `Gemini_Generated_Image_3me5nd3me5nd3me5_clean_sq512.png` : Image normalisée carrée 512×512 avec fond blanc et marge de sécurité de 15%.

## 🚀 Restauration Rapide
Pour recharger ce modèle directement dans l'interface web :
* Visualiseur Three.js : Ouvrir `http://127.0.0.1:8080/outputs/version_clown/model_bambu.glb`
* Slicer Bambu Lab : Glisser-déposer `model_bambu.stl` directement dans Bambu Studio.
