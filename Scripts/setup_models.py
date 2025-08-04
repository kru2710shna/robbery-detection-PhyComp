import os
from pathlib import Path
import urllib.request


def download_yolov8s_model():
    model_dir = Path("../models/yolo_weights/")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "yolov8s.pt"

    if not model_path.exists():
        print("⬇️ Downloading yolov8s.pt model weights...")
        url = (
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt"
        )
        urllib.request.urlretrieve(url, model_path)
        print("✅ Download complete:", model_path)
    else:
        print("✅ yolov8m.pt already exists.")


def download_yolov8s_pose_model():
    url = (
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-pose.pt"
    )
    save_dir = "models/yolo_weights"
    save_path = os.path.join(save_dir, "yolov8s-pose.pt")

    os.makedirs(save_dir, exist_ok=True)
    if not os.path.exists(save_path):
        print("📥 Downloading YOLOv8s-Pose model...")
        urllib.request.urlretrieve(url, save_path)
        print("✅ Download complete.")
    else:
        print("✔️ YOLOv8s-Pose model already exists at:", save_path)


def download_weapon_model():
    model_path = "models/weapon_model.pt"
    if os.path.exists(model_path):
        print("✅ Weapon detection model already exists.")
        return
    try:
        print("📥 Downloading pretrained weapon detection model...")
        url = "https://example.com/path/to/weapon_model.pt" 
        urllib.request.urlretrieve(url, model_path)
        print("✅ Download complete.")
    except Exception as e:
        print(f"❌ Failed to download weapon model: {e}")



def verify_weapon_dataset():
    dataset_dir = Path("../datasets/weapon_detection/")
    if dataset_dir.exists() and (dataset_dir / "data.yaml").exists():
        print("✅ Weapon detection dataset already organized and available.")
    else:
        print("⚠️ Please make sure the weapon detection dataset is manually downloaded and placed at:", dataset_dir)
        
        

if __name__ == "__main__":
    download_yolov8s_model()
    download_yolov8s_pose_model()
    # verify_weapon_dataset()
    # download_weapon_model()
