#!/bin/bash
set -e

MODELS_DIR="./models"
mkdir -p "$MODELS_DIR"/{mimic_motion,svd,dwpose,codeformer,sam}

echo "============================================"
echo "  Palmera Live - Downloading Model Weights"
echo "============================================"

# 1. MimicMotion (~3GB)
echo ""
echo "[1/5] Downloading MimicMotion..."
if [ ! -f "$MODELS_DIR/mimic_motion/MimicMotion_1-1.pth" ]; then
    huggingface-cli download tencent/MimicMotion \
        --local-dir "$MODELS_DIR/mimic_motion" \
        --include "MimicMotion_1-1.pth"
    echo "  -> MimicMotion downloaded."
else
    echo "  -> MimicMotion already exists, skipping."
fi

# 2. Stable Video Diffusion 1.1 (~10GB)
echo ""
echo "[2/5] Downloading SVD 1.1..."
if [ ! -d "$MODELS_DIR/svd/unet" ]; then
    huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt-1-1 \
        --local-dir "$MODELS_DIR/svd"
    echo "  -> SVD 1.1 downloaded."
else
    echo "  -> SVD 1.1 already exists, skipping."
fi

# 3. DWPose (~400MB)
echo ""
echo "[3/5] Downloading DWPose..."
if [ ! -f "$MODELS_DIR/dwpose/dw-ll_ucoco_384.onnx" ]; then
    wget -q --show-progress -O "$MODELS_DIR/dwpose/dw-ll_ucoco_384.onnx" \
        "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx"
    wget -q --show-progress -O "$MODELS_DIR/dwpose/yolox_l.onnx" \
        "https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx"
    echo "  -> DWPose downloaded."
else
    echo "  -> DWPose already exists, skipping."
fi

# 4. CodeFormer (~400MB)
echo ""
echo "[4/5] Downloading CodeFormer..."
if [ ! -f "$MODELS_DIR/codeformer/codeformer.pth" ]; then
    wget -q --show-progress -O "$MODELS_DIR/codeformer/codeformer.pth" \
        "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
    # Face detection model needed by CodeFormer
    mkdir -p "$MODELS_DIR/codeformer/facelib"
    wget -q --show-progress -O "$MODELS_DIR/codeformer/facelib/detection_Resnet50_Final.pth" \
        "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth"
    wget -q --show-progress -O "$MODELS_DIR/codeformer/facelib/parsing_parsenet.pth" \
        "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth"
    echo "  -> CodeFormer downloaded."
else
    echo "  -> CodeFormer already exists, skipping."
fi

# 5. SAM ViT-B (~375MB)
echo ""
echo "[5/5] Downloading SAM (Segment Anything)..."
if [ ! -f "$MODELS_DIR/sam/sam_vit_b.pth" ]; then
    wget -q --show-progress -O "$MODELS_DIR/sam/sam_vit_b.pth" \
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    echo "  -> SAM downloaded."
else
    echo "  -> SAM already exists, skipping."
fi

echo ""
echo "============================================"
echo "  All weights downloaded successfully!"
echo "  Total disk usage:"
du -sh "$MODELS_DIR"
echo "============================================"
