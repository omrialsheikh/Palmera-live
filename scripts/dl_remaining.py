"""Download remaining model weights using Python (no shell issues)."""
import urllib.request
import os

downloads = [
    ("./models/codeformer/facelib/detection_Resnet50_Final.pth", "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth"),
    ("./models/codeformer/facelib/parsing_parsenet.pth", "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth"),
    ("./models/sam/sam_vit_b.pth", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"),
]

for dest, url in downloads:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        print(f"[skip] {dest}")
        continue
    print(f"[downloading] {dest}...")
    urllib.request.urlretrieve(url, dest)
    size_mb = os.path.getsize(dest) / 1024 / 1024
    print(f"[done] {dest} ({size_mb:.0f} MB)")

print("\nAll done!")
