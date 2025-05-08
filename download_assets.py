import os
import urllib.request

BASE_DIR = os.getcwd()
MODELS_DIR = os.path.join(BASE_DIR, "models")

def ensure_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)

def download_file(url, dest):
    if not os.path.exists(dest):
        print(f"🔽 Downloading {os.path.basename(dest)}...")
        urllib.request.urlretrieve(url, dest)
        print(f"✅ Downloaded: {os.path.basename(dest)}")
    else:
        print(f"⏭️ Already exists: {os.path.basename(dest)}")

def download_all_assets():
    ensure_dirs()

    download_file(
        "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10m.pt",
        os.path.join(MODELS_DIR, "yolov10m.pt")
    )

    deeplabv3_url = "https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth"
    deeplabv3_dest = os.path.join(MODELS_DIR, "deeplabv3_resnet101_coco-586e9e4e.pth")
    download_file(deeplabv3_url, deeplabv3_dest)
