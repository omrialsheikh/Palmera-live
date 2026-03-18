"""
Palmera Live - Sanity Check
Verifies GPU, CUDA, and model weights before running the pipeline.
"""

import sys
import os

def check_gpu():
    print("=" * 50)
    print("  GPU & CUDA Check")
    print("=" * 50)

    import torch

    if not torch.cuda.is_available():
        print("  CUDA: NOT AVAILABLE")
        print("  -> Cannot run pipeline without GPU!")
        return False

    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.version.cuda}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Quick VRAM test
    try:
        x = torch.randn(1, 4, 128, 96, device="cuda", dtype=torch.float16)
        del x
        torch.cuda.empty_cache()
        print("  VRAM allocation: OK")
    except Exception as e:
        print(f"  VRAM allocation: FAILED ({e})")
        return False

    return True


def check_models():
    print("\n" + "=" * 50)
    print("  Model Weights Check")
    print("=" * 50)

    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    models_dir = os.path.abspath(models_dir)

    required = {
        "MimicMotion": os.path.join(models_dir, "mimic_motion", "MimicMotion_1-1.pth"),
        "SVD (UNet)": os.path.join(models_dir, "svd", "unet", "diffusion_pytorch_model.fp16.safetensors"),
        "DWPose": os.path.join(models_dir, "dwpose", "dw-ll_ucoco_384.onnx"),
        "DWPose (YOLOX)": os.path.join(models_dir, "dwpose", "yolox_l.onnx"),
        "CodeFormer": os.path.join(models_dir, "codeformer", "codeformer.pth"),
        "SAM": os.path.join(models_dir, "sam", "sam_vit_b.pth"),
    }

    all_ok = True
    for name, path in required.items():
        exists = os.path.exists(path)
        size = ""
        if exists:
            size_mb = os.path.getsize(path) / 1024 / 1024
            size = f" ({size_mb:.0f} MB)"
        status = "OK" + size if exists else "MISSING"
        symbol = "+" if exists else "!"
        print(f"  [{symbol}] {name}: {status}")
        if not exists:
            all_ok = False

    return all_ok


def check_imports():
    print("\n" + "=" * 50)
    print("  Dependencies Check")
    print("=" * 50)

    deps = [
        ("torch", "PyTorch"),
        ("diffusers", "Diffusers"),
        ("transformers", "Transformers"),
        ("cv2", "OpenCV"),
        ("fastapi", "FastAPI"),
        ("segment_anything", "SAM"),
        ("controlnet_aux", "ControlNet Aux (DWPose)"),
        ("basicsr", "BasicSR (CodeFormer)"),
        ("einops", "Einops"),
        ("omegaconf", "OmegaConf"),
    ]

    all_ok = True
    for module, name in deps:
        try:
            __import__(module)
            print(f"  [+] {name}")
        except ImportError:
            print(f"  [!] {name}: NOT INSTALLED")
            all_ok = False

    return all_ok


def main():
    print("\n")
    print("  PALMERA LIVE - Sanity Check")
    print("\n")

    gpu_ok = check_gpu()
    imports_ok = check_imports()
    models_ok = check_models()

    print("\n" + "=" * 50)
    print("  Summary")
    print("=" * 50)
    print(f"  GPU:          {'PASS' if gpu_ok else 'FAIL'}")
    print(f"  Dependencies: {'PASS' if imports_ok else 'FAIL'}")
    print(f"  Models:       {'PASS' if models_ok else 'FAIL'}")

    if gpu_ok and imports_ok and models_ok:
        print("\n  All checks passed! Ready to run.")
        return 0
    else:
        print("\n  Some checks failed. Fix issues above before running.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
