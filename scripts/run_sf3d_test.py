import os
import sys
import time

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

from gradio_client import Client, handle_file
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs", "sota_test")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

head_img = os.path.join(OUTPUTS_DIR, "van_aert_head_2d_reference.png")
full_img = os.path.join(BASE_DIR, "outputs", "van_aert_cleaned.png")

client = Client("stabilityai/stable-fast-3d")

print("--- Testing SF3D on Head Crop ---")
t0 = time.time()
try:
    res_head = client.predict(
        input_image=handle_file(head_img),
        foreground_ratio=0.90,
        remesh_option="None",
        vertex_count=-1,
        texture_size=1024,
        api_name="/run_button"
    )
    print(f"SF3D Head generated in {time.time()-t0:.2f}s! Output:", res_head)
    if isinstance(res_head, (list, tuple)) and len(res_head) > 1:
        model_path = res_head[1]
        out_glb = os.path.join(OUTPUTS_DIR, "sf3d_head.glb")
        shutil.copy2(model_path, out_glb)
        print("Copied head GLB to:", out_glb, "Size:", os.path.getsize(out_glb))
except Exception as e:
    print("SF3D Head Error:", e)
