#!/bin/bash
set -e

THIRD_PARTY_DIR="./third_party"
mkdir -p "$THIRD_PARTY_DIR"

echo "============================================"
echo "  Setting up MimicMotion"
echo "============================================"

if [ ! -d "$THIRD_PARTY_DIR/MimicMotion" ]; then
    echo "[1/2] Cloning MimicMotion repository..."
    git clone https://github.com/tencent/MimicMotion.git "$THIRD_PARTY_DIR/MimicMotion"
    echo "  -> Cloned."
else
    echo "[1/2] MimicMotion already cloned, pulling latest..."
    cd "$THIRD_PARTY_DIR/MimicMotion" && git pull && cd -
fi

echo "[2/2] Installing MimicMotion dependencies..."
pip install -q onnxruntime-gpu

echo ""
echo "============================================"
echo "  MimicMotion setup complete!"
echo "============================================"
