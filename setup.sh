#!/bin/bash
# ============================================================
# setup.sh — Run this once on Lightning.ai Studio to prepare
# the environment before launching app.py
# ============================================================
set -e

echo "=== [1/5] Installing Python dependencies ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 -q
pip install -r requirements.txt -q

echo "=== [2/5] Installing basicsr from local source ==="
cd basicsr-1.4.2
pip install -e . -q
cd ..

echo "=== [3/5] Patching gradio_client for Python 3.13 compatibility ==="
UTILS=$(python -c "import gradio_client.utils as u; import inspect; print(inspect.getfile(u))")
# Patch get_type to handle bool schema
python - <<'PYEOF'
import re, pathlib
path = pathlib.Path(__import__('gradio_client').utils.__file__)
src = path.read_text()
old = "def get_type(schema: dict):\n    if \"const\" in schema:"
new = "def get_type(schema: dict):\n    if isinstance(schema, bool):\n        return \"Any\"\n    if \"const\" in schema:"
if old in src:
    path.write_text(src.replace(old, new))
    print("  get_type patched OK")
old2 = "    if schema == {}:\n        return \"Any\""
new2 = "    if isinstance(schema, bool) or schema == {}:\n        return \"Any\""
src = path.read_text()
if old2 in src:
    path.write_text(src.replace(old2, new2))
    print("  _json_schema_to_python_type patched OK")
PYEOF

echo "=== [4/5] Downloading model weights ==="
mkdir -p weights weights/CodeFormer weights/facelib

# RealESRGAN
[ -f weights/RealESRGAN_x2.pth ] || curl -L -# "https://huggingface.co/sberbank-ai/Real-ESRGAN/resolve/main/RealESRGAN_x2.pth" -o weights/RealESRGAN_x2.pth
[ -f weights/RealESRGAN_x4.pth ] || curl -L -# "https://huggingface.co/sberbank-ai/Real-ESRGAN/resolve/main/RealESRGAN_x4.pth" -o weights/RealESRGAN_x4.pth
[ -f weights/RealESRGAN_x8.pth ] || curl -L -# "https://huggingface.co/sberbank-ai/Real-ESRGAN/resolve/main/RealESRGAN_x8.pth" -o weights/RealESRGAN_x8.pth

# RetinaFace + ParseNet (for facexlib)
FACELIB_DIR=$(python -c "import facexlib; import os; print(os.path.join(os.path.dirname(facexlib.__file__), 'weights'))")
mkdir -p "$FACELIB_DIR"
[ -f "$FACELIB_DIR/detection_Resnet50_Final.pth" ] || curl -L -k -# "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth" -o "$FACELIB_DIR/detection_Resnet50_Final.pth"
[ -f "$FACELIB_DIR/parsing_parsenet.pth" ] || curl -L -k -# "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth" -o "$FACELIB_DIR/parsing_parsenet.pth"

# CodeFormer — clone repo & download weights
if [ ! -d "third_party/CodeFormer" ]; then
    mkdir -p third_party
    git clone --depth 1 https://github.com/sczhou/CodeFormer.git third_party/CodeFormer
fi

if [ ! -f "weights/codeformer.pth" ]; then
    cd third_party/CodeFormer
    pip install -r requirements.txt -q
    python scripts/download_pretrained_models.py CodeFormer
    python scripts/download_pretrained_models.py facelib
    cd ../..
    cp third_party/CodeFormer/weights/CodeFormer/codeformer.pth weights/codeformer.pth
    mkdir -p weights/facelib
    cp third_party/CodeFormer/weights/facelib/*.pth weights/facelib/ 2>/dev/null || true
fi

echo "=== [5/5] Setup complete! Run: python app.py ==="
