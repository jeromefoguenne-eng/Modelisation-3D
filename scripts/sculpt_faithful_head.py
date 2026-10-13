import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import map_coordinates, gaussian_filter

# 1. Load pure isolated TripoSR head
head_path = "outputs/sota_test/pure_isolated_head_triposr.glb"
m_head = trimesh.load(head_path, force="mesh")
print(f"Loaded head mesh: {len(m_head.vertices)} vertices, {len(m_head.faces)} faces")

# 2. Load Depth Map & Reference Image
depth_img = Image.open("outputs/sota_test/head_depth_v2.png").convert("L")
ref_img = Image.open("outputs/sota_test/pure_head_full_helmet.png").convert("RGBA")

d_arr = np.array(depth_img, dtype=np.float32) / 255.0
ref_arr = np.array(ref_img)
H, W = d_arr.shape

# Clean mask of the subject in 2D
mask_2d = ref_arr[:, :, 3] > 20

# High-pass and feature enhancement on depth map:
# Mouth cavity detection: mouth is in lower half of face (y between 300 and 370, x between 140 and 220)
# In ref_arr, mouth pixels have darker pink/red color
rgb = ref_arr[:, :, :3]
# Oral cavity mask
is_mouth = (np.arange(H)[:, None] >= 290) & (np.arange(H)[:, None] <= 375) & \
           (np.arange(W)[None, :] >= 130) & (np.arange(W)[None, :] <= 230) & \
           (rgb[:, :, 0] > 100) & (rgb[:, :, 1] < 120) & (rgb[:, :, 2] < 120)

# Visor mask (glasses): row 200 to 270, x between 100 and 270
is_visor = (np.arange(H)[:, None] >= 200) & (np.arange(H)[:, None] <= 280) & \
           (np.arange(W)[None, :] >= 90) & (np.arange(W)[None, :] <= 280)

# Chin mask: row 365 to 400, x between 145 and 215
is_chin = (np.arange(H)[:, None] >= 365) & (np.arange(H)[:, None] <= 400) & \
          (np.arange(W)[None, :] >= 145) & (np.arange(W)[None, :] <= 215)

# Build enhanced sculpted depth map
# Base depth variation
enhanced_d = d_arr.copy()
# Deepen oral cavity by subtracting depth inside mouth
mouth_cavity_depth = gaussian_filter(is_mouth.astype(np.float32), sigma=4.0)
enhanced_d -= 0.35 * mouth_cavity_depth

# Enhance visor protrusion
visor_protrusion = gaussian_filter(is_visor.astype(np.float32), sigma=4.0)
enhanced_d += 0.25 * visor_protrusion

# Enhance chin protrusion
chin_protrusion = gaussian_filter(is_chin.astype(np.float32), sigma=3.0)
enhanced_d += 0.20 * chin_protrusion

# Normalize
enhanced_d = (enhanced_d - enhanced_d.min()) / (enhanced_d.max() - enhanced_d.min() + 1e-6)

# 3. Apply Sculpting to Mesh Vertices
# Normalize head vertices to standard bounding box
b = m_head.bounds
# Target head size: height = 26.0 mm (from neck base to helmet top)
scale_factor = 26.0 / m_head.extents[1]
m_head.apply_scale(scale_factor)
b = m_head.bounds
# Center X and Z
m_head.apply_translation([-(b[0][0] + b[1][0]) / 2.0, -b[0][1], -(b[0][2] + b[1][2]) / 2.0])

v = m_head.vertices.copy()
normals = m_head.vertex_normals

# Map 3D X, Y onto 2D (U, V)
# Bounding box in X: [-10, 10], in Y: [0, 26]
# In 2D reference, subject bounds:
ys, xs = np.where(mask_2d)
u_min, u_max = xs.min(), xs.max()
v_min, v_max = ys.min(), ys.max()

# X normalized to [0, 1]
# Notice in front view: positive X is subject's left (viewer's right)
# Let's align 3D X with 2D X:
x_span = v[:, 0].max() - v[:, 0].min()
y_span = v[:, 1].max() - v[:, 1].min()

u_coords = u_min + (v[:, 0] - v[:, 0].min()) / x_span * (u_max - u_min)
# Y is inverted (top of head Y=26 is top of image v_min)
v_coords = v_max - (v[:, 1] - v[:, 1].min()) / y_span * (v_max - v_min)

u_coords = np.clip(u_coords, 0, W - 1)
v_coords = np.clip(v_coords, 0, H - 1)

# Sample enhanced depth
depth_samples = map_coordinates(enhanced_d, [v_coords, u_coords], order=1, mode='nearest')

# Sculpting factor:
# Apply only to front-facing surfaces (normals[:, 2] > 0.15)
facing_forward = np.clip((normals[:, 2] - 0.15) / 0.7, 0.0, 1.0)

# Organic cranial curve: natural face rounds back towards ears
# Center of face is X=0
radius_x = 9.5
cranial_arch = np.maximum(0.0, 1.0 - (v[:, 0] / radius_x) ** 2)

# Total 3D displacement along Z (forward/backward)
# Total depth range: 4.5 mm
z_displacement = facing_forward * (depth_samples - 0.5) * 4.2 * (0.6 + 0.4 * cranial_arch)

# Apply displacement
v[:, 2] += z_displacement

# Update mesh
m_head.vertices = v
m_head.fix_normals()

# Smooth slightly with Taubin to remove any discretization artifacts
trimesh.smoothing.filter_taubin(m_head, lamb=0.35, nu=-0.37, iterations=5)

out_glb = "outputs/sota_test/sculpted_faithful_head.glb"
m_head.export(out_glb)
print("Saved sculpted head:", out_glb)

# Render clay preview
from core.studio_renderer import render_mesh_preview
clay_mesh = trimesh.Trimesh(vertices=m_head.vertices, faces=m_head.faces, process=False)
clay_mesh.visual.vertex_colors = [185, 185, 190, 255]
render_mesh_preview(clay_mesh, "outputs/sota_test/sculpted_faithful_head_clay.png", img_size=1024, angle_deg=22.0)
print("Rendered sculpted_faithful_head_clay.png!")
