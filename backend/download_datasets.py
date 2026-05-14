"""
OMNIVIS - Dataset Download Script
Downloads real-world datasets for training face detection and medical imaging models.
Sources: Kaggle, HuggingFace, direct URLs, and public datasets.
"""
import os
import sys
import shutil
import requests
import zipfile
import tarfile
import logging
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("datasets")

DATA_DIR = Path(__file__).parent / "datasets"
MODELS_DIR = Path(__file__).parent / "models"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> Path:
    """Download a file with progress bar."""
    if dest.exists():
        logger.info(f"Already exists: {dest}")
        return dest

    logger.info(f"Downloading {url} -> {dest}")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        desc=dest.name,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))

    return dest


def extract_archive(archive_path: Path, dest_dir: Path) -> Path:
    """Extract zip/tar.gz archives."""
    if dest_dir.exists() and list(dest_dir.iterdir()):
        logger.info(f"Already extracted: {dest_dir}")
        return dest_dir

    logger.info(f"Extracting {archive_path} -> {dest_dir}")
    os.makedirs(dest_dir, exist_ok=True)

    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(dest_dir)
    elif archive_path.suffix == ".gz":
        with tarfile.open(archive_path, "r:gz") as t:
            t.extractall(dest_dir)

    return dest_dir


def download_fer2013():
    """
    FER2013 - Facial Expression Recognition dataset.
    35,685 48x48 grayscale images of faces with 7 emotion labels.
    Labels: angry, disgust, fear, happy, sad, surprise, neutral
    """
    logger.info("=" * 60)
    logger.info("Downloading FER2013 Dataset")
    logger.info("=" * 60)

    fer_dir = DATA_DIR / "fer2013"
    os.makedirs(fer_dir, exist_ok=True)

    csv_url = "https://raw.githubusercontent.com/josephmisiti/awesome-machine-learning/master/datasets/fer2013.csv"
    csv_path = fer_dir / "fer2013.csv"

    try:
        download_file(csv_url, csv_path)

        import pandas as pd
        import cv2
        import numpy as np

        logger.info("Processing FER2013 CSV into image files...")
        df = pd.read_csv(csv_path)

        split_dirs = {"Training": fer_dir / "train", "PublicTest": fer_dir / "val", "PrivateTest": fer_dir / "test"}
        for d in split_dirs.values():
            os.makedirs(d, exist_ok=True)

        emotion_labels = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "sad", 5: "surprise", 6: "neutral"}

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Converting pixels to images"):
            pixels = row["pixels"].split()
            image = np.array(pixels, dtype=np.uint8).reshape(48, 48)

            emotion = int(row["emotion"])
            usage = row["Usage"]

            emotion_dir = split_dirs[usage] / emotion_labels[emotion]
            os.makedirs(emotion_dir, exist_ok=True)

            filename = f"{usage}_{row.name}.png"
            cv2.imwrite(str(emotion_dir / filename), image)

        logger.info(f"FER2013 saved to {fer_dir}")
        return fer_dir
    except Exception as e:
        logger.error(f"FER2013 download failed: {e}")
        logger.info("FER2013 can also be downloaded from Kaggle:")
        logger.info("  kaggle datasets download -d msambare/fer2013")
        return None


def download_wider_face():
    """
    WIDER FACE - Face detection dataset.
    32,203 images with 393,703 annotated faces.
    """
    logger.info("=" * 60)
    logger.info("Downloading WIDER FACE Dataset")
    logger.info("=" * 60)

    wider_dir = DATA_DIR / "wider_face"
    os.makedirs(wider_dir, exist_ok=True)

    logger.info("WIDER FACE is available from:")
    logger.info("  http://shuoyang1213.me/WIDERFACE/")
    logger.info("  Or via Kaggle: kaggle datasets download -d xianfan/wider-face")
    logger.info("  Or direct: http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip")

    try:
        annotations_url = "http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip"
        zip_path = wider_dir / "wider_face_split.zip"
        download_file(annotations_url, zip_path)
        extract_archive(zip_path, wider_dir)
        logger.info(f"WIDER FACE annotations saved to {wider_dir}")
    except Exception as e:
        logger.error(f"WIDER FACE annotations download failed: {e}")

    return wider_dir


def download_chest_xray():
    """
    Chest X-Ray Images (Pneumonia) from Kaggle.
    5,863 X-Ray images (JPEG) with pneumonia/normal labels.
    """
    logger.info("=" * 60)
    logger.info("Downloading Chest X-Ray Dataset")
    logger.info("=" * 60)

    xray_dir = DATA_DIR / "chest_xray"
    os.makedirs(xray_dir, exist_ok=True)

    logger.info("Chest X-Ray dataset is available on Kaggle:")
    logger.info("  kaggle datasets download -d paultimothymooney/chest-xray-pneumonia")
    logger.info("After downloading, extract to:")
    logger.info(f"  {xray_dir}")

    logger.info("Alternative: NIH Chest X-ray dataset (112,120 images)")
    logger.info("  https://nihcc.app.box.com/v/ChestXray-NIHCC")

    try:
        logger.info("Downloading sample chest X-ray images for testing...")

        sample_images = [
            ("https://github.com/ieee8023/covid-chestxray-dataset/raw/master/images/1-s2.0-S1684118220300682-main.pdf-001-a.png", "sample_normal.png"),
        ]

        for url, filename in sample_images:
            dest = xray_dir / filename
            try:
                download_file(url, dest)
            except Exception:
                logger.warning(f"Could not download sample: {url}")

        logger.info(f"Sample images saved to {xray_dir}")
    except Exception as e:
        logger.error(f"Chest X-Ray download failed: {e}")

    return xray_dir


def download_medical_scan_classification():
    """
    Medical Scan Classification Dataset (X-Ray, MRI, CT, Ultrasound).
    For training the scan type classifier.
    """
    logger.info("=" * 60)
    logger.info("Downloading Medical Scan Classification Dataset")
    logger.info("=" * 60)

    medical_dir = DATA_DIR / "medical_scans"
    scan_types = ["x-ray", "mri", "ct", "ultrasound"]
    for st in scan_types:
        os.makedirs(medical_dir / "train" / st, exist_ok=True)
        os.makedirs(medical_dir / "val" / st, exist_ok=True)
        os.makedirs(medical_dir / "test" / st, exist_ok=True)

    logger.info("Medical scan datasets are available from:")
    logger.info("  - MURA (Musculoskeletal X-Rays): https://stanfordmlgroup.github.io/competitions/mura/")
    logger.info("  - BraTS (Brain MRI): https://www.med.upenn.edu/cbica/brats/")
    logger.info("  - LIDC-IDRI (Lung CT): https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254")
    logger.info("  - BUSI (Breast Ultrasound): https://scholar.cu.edu.eg/?q=afahmy/pages/dataset")
    logger.info(f"\nOrganize downloaded images into: {medical_dir}/train/<scan_type>/")
    logger.info(f"  {medical_dir}/val/<scan_type>/")
    logger.info(f"  {medical_dir}/test/<scan_type>/")

    return medical_dir


def download_face_attributes():
    """
    CelebA - Face attributes dataset.
    202,599 face images with 40 attribute annotations.
    """
    logger.info("=" * 60)
    logger.info("Downloading CelebA Dataset")
    logger.info("=" * 60)

    celeb_dir = DATA_DIR / "celeba"
    os.makedirs(celeb_dir, exist_ok=True)

    logger.info("CelebA is available from:")
    logger.info("  - Google Drive: https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg")
    logger.info("  - Kaggle: kaggle datasets download -d jessicali9530/celeba-dataset")
    logger.info("After downloading, extract to:")
    logger.info(f"  {celeb_dir}")

    return celeb_dir


def download_imagenet_subset():
    """
    Download ImageNet subset for pre-trained model evaluation.
    """
    logger.info("=" * 60)
    logger.info("ImageNet Subset for Evaluation")
    logger.info("=" * 60)

    imagenet_dir = DATA_DIR / "imagenet_subset"
    os.makedirs(imagenet_dir, exist_ok=True)

    logger.info("ImageNet can be downloaded from:")
    logger.info("  - Official: https://image-net.org/download.php")
    logger.info("  - Tiny ImageNet: http://cs231n.stanford.edu/tiny-imagenet-200.zip")

    try:
        tiny_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        zip_path = imagenet_dir / "tiny-imagenet-200.zip"
        download_file(tiny_url, zip_path)
        extract_archive(zip_path, imagenet_dir)
        logger.info(f"Tiny ImageNet saved to {imagenet_dir}")
    except Exception as e:
        logger.error(f"Tiny ImageNet download failed: {e}")

    return imagenet_dir


def download_pretrained_weights():
    """Download pre-trained model weights."""
    logger.info("=" * 60)
    logger.info("Downloading Pre-trained Weights")
    logger.info("=" * 60)

    models_urls = {
        "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        "res10_300x300_ssd_iter_140000.caffemodel": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
    }

    for filename, url in models_urls.items():
        dest = MODELS_DIR / filename
        try:
            download_file(url, dest)
            logger.info(f"Downloaded: {filename}")
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")


def download_kaggle_dataset(dataset_name: str, output_dir: Path):
    """Download a dataset from Kaggle using the Kaggle API."""
    try:
        import kaggle
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Downloading Kaggle dataset: {dataset_name}")
        kaggle.api.dataset_download_files(dataset_name, path=str(output_dir), unzip=True)
        logger.info(f"Downloaded to: {output_dir}")
        return True
    except Exception as e:
        logger.error(f"Kaggle download failed: {e}")
        logger.info("Make sure you have Kaggle API credentials set:")
        logger.info("  1. Create kaggle.json from https://www.kaggle.com/account")
        logger.info("  2. Place it in ~/.kaggle/kaggle.json")
        logger.info("  3. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False


def create_synthetic_dataset():
    """Create a small synthetic dataset for testing training pipeline."""
    logger.info("=" * 60)
    logger.info("Creating Synthetic Test Dataset")
    logger.info("=" * 60)

    import cv2
    import numpy as np

    synthetic_dir = DATA_DIR / "synthetic"
    train_dir = synthetic_dir / "train"
    val_dir = synthetic_dir / "val"

    for split_dir in [train_dir, val_dir]:
        for emotion in ["happy", "sad", "neutral", "angry", "surprise"]:
            os.makedirs(split_dir / emotion, exist_ok=True)
        for scan_type in ["x-ray", "mri", "ct", "ultrasound"]:
            os.makedirs(split_dir / scan_type, exist_ok=True)

    num_per_class_train = 50
    num_per_class_val = 10

    np.random.seed(42)

    for split_dir, num in [(train_dir, num_per_class_train), (val_dir, num_per_class_val)]:
        for i in range(num):
            for emotion in ["happy", "sad", "neutral", "angry", "surprise"]:
                img = _generate_synthetic_face(emotion, 48)
                cv2.imwrite(str(split_dir / emotion / f"{emotion}_{i}.png"), img)

            for scan_type in ["x-ray", "mri", "ct", "ultrasound"]:
                img = _generate_synthetic_scan(scan_type, 224)
                cv2.imwrite(str(split_dir / scan_type / f"{scan_type}_{i}.png"), img)

    logger.info(f"Synthetic dataset created at: {synthetic_dir}")
    logger.info(f"  Train: {num_per_class_train} images per class")
    logger.info(f"  Val: {num_per_class_val} images per class")

    return synthetic_dir


def _generate_synthetic_face(emotion: str, size: int = 48) -> np.ndarray:
    """Generate a synthetic face image for testing."""
    img = np.zeros((size, size), dtype=np.uint8)

    cx, cy = size // 2, size // 2
    cv2.ellipse(img, (cx, cy), (size//3, size//2), 0, 0, 360, 180, -1)

    eye_y = size // 3
    cv2.circle(img, (cx - size//6, eye_y), size//12, 255, -1)
    cv2.circle(img, (cx + size//6, eye_y), size//12, 255, -1)
    cv2.circle(img, (cx - size//6, eye_y), size//20, 0, -1)
    cv2.circle(img, (cx + size//6, eye_y), size//20, 0, -1)

    mouth_y = size * 2 // 3
    if emotion == "happy":
        cv2.ellipse(img, (cx, mouth_y), (size//5, size//8), 0, 0, 180, 255, 2)
    elif emotion == "sad":
        cv2.ellipse(img, (cx, mouth_y + 5), (size//5, size//8), 0, 180, 360, 255, 2)
    elif emotion == "surprise":
        cv2.ellipse(img, (cx, mouth_y), (size//8, size//6), 0, 0, 360, 255, -1)
    elif emotion == "angry":
        cv2.line(img, (cx - size//4, eye_y - size//6), (cx - size//10, eye_y - size//10), 255, 2)
        cv2.line(img, (cx + size//4, eye_y - size//6), (cx + size//10, eye_y - size//10), 255, 2)
        cv2.rectangle(img, (cx - size//6, mouth_y - size//12), (cx + size//6, mouth_y + size//12), 255, 2)
    else:
        cv2.line(img, (cx - size//6, mouth_y), (cx + size//6, mouth_y), 255, 2)

    noise = np.random.normal(0, 10, (size, size)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def _generate_synthetic_scan(scan_type: str, size: int = 224) -> np.ndarray:
    """Generate a synthetic medical scan image for testing."""
    img = np.zeros((size, size), dtype=np.uint8)

    if scan_type == "x-ray":
        cv2.rectangle(img, (size//4, size//6), (3*size//4, 5*size//6), 100, -1)
        cv2.ellipse(img, (size//3, size//2), (size//6, size//4), 0, 0, 360, 150, -1)
        cv2.ellipse(img, (2*size//3, size//2), (size//6, size//4), 0, 0, 360, 150, -1)
        cv2.rectangle(img, (size//2 - 10, size//4), (size//2 + 10, 3*size//4), 180, -1)
    elif scan_type == "mri":
        cv2.ellipse(img, (size//2, size//2), (size//3, size//3), 0, 0, 360, 120, -1)
        cv2.ellipse(img, (size//2, size//2), (size//4, size//4), 0, 0, 360, 80, -1)
        cv2.ellipse(img, (size//2, size//2), (size//6, size//6), 0, 0, 360, 160, -1)
        for _ in range(5):
            rx, ry = np.random.randint(size//4, 3*size//4, 2)
            cv2.circle(img, (int(rx), int(ry)), np.random.randint(5, 15), np.random.randint(100, 200), -1)
    elif scan_type == "ct":
        cv2.ellipse(img, (size//2, size//2), (size//3, size//3), 0, 0, 360, 100, -1)
        for _ in range(20):
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, size//3)
            x = int(size//2 + r * np.cos(angle))
            y = int(size//2 + r * np.sin(angle))
            cv2.circle(img, (x, y), np.random.randint(2, 8), np.random.randint(50, 200), -1)
    elif scan_type == "ultrasound":
        for y in range(0, size, 2):
            for x in range(0, size, 2):
                if np.random.random() > 0.3:
                    img[y, y % size] = np.random.randint(50, 200)
        cv2.ellipse(img, (size//2, size//2), (size//5, size//4), 0, 0, 360, 180, -1)

    noise = np.random.normal(0, 15, (size, size)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def print_summary():
    """Print summary of available datasets."""
    logger.info("=" * 60)
    logger.info("Dataset Summary")
    logger.info("=" * 60)

    datasets = {
        "FER2013": DATA_DIR / "fer2013",
        "WIDER FACE": DATA_DIR / "wider_face",
        "Chest X-Ray": DATA_DIR / "chest_xray",
        "Medical Scans": DATA_DIR / "medical_scans",
        "CelebA": DATA_DIR / "celeba",
        "ImageNet Subset": DATA_DIR / "imagenet_subset",
        "Synthetic Test": DATA_DIR / "synthetic",
    }

    for name, path in datasets.items():
        if path.exists():
            file_count = sum(1 for _ in path.rglob("*") if _.is_file())
            logger.info(f" {name}: {path} ({file_count} files)")
        else:
            logger.info(f"  {name}: Not downloaded")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download datasets for OMNIVIS training")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--fer2013", action="store_true", help="Download FER2013 emotion dataset")
    parser.add_argument("--wider-face", action="store_true", help="Download WIDER FACE dataset")
    parser.add_argument("--chest-xray", action="store_true", help="Download Chest X-Ray dataset")
    parser.add_argument("--medical-scans", action="store_true", help="Download medical scan dataset")
    parser.add_argument("--celeba", action="store_true", help="Download CelebA dataset")
    parser.add_argument("--imagenet", action="store_true", help="Download Tiny ImageNet")
    parser.add_argument("--weights", action="store_true", help="Download pre-trained weights")
    parser.add_argument("--synthetic", action="store_true", help="Create synthetic test dataset")
    parser.add_argument("--kaggle", type=str, help="Download from Kaggle (dataset name)")
    parser.add_argument("--summary", action="store_true", help="Show dataset summary")

    args = parser.parse_args()

    if args.summary:
        print_summary()
        return

    if not any([args.all, args.fer2013, args.wider_face, args.chest_xray,
                args.medical_scans, args.celeba, args.imagenet, args.weights,
                args.synthetic, args.kaggle]):
        logger.info("No dataset specified. Creating synthetic dataset for testing...")
        download_synthetic_dataset()
        print_summary()
        return

    if args.all:
        download_fer2013()
        download_wider_face()
        download_chest_xray()
        download_medical_scan_classification()
        download_face_attributes()
        download_imagenet_subset()
        download_pretrained_weights()
        download_synthetic_dataset()
    else:
        if args.fer2013:
            download_fer2013()
        if args.wider_face:
            download_wider_face()
        if args.chest_xray:
            download_chest_xray()
        if args.medical_scans:
            download_medical_scan_classification()
        if args.celeba:
            download_face_attributes()
        if args.imagenet:
            download_imagenet_subset()
        if args.weights:
            download_pretrained_weights()
        if args.synthetic:
            download_synthetic_dataset()
        if args.kaggle:
            download_kaggle_dataset(args.kaggle, DATA_DIR / args.kaggle.split("/")[-1])

    print_summary()


if __name__ == "__main__":
    main()
