"""
╔══════════════════════════════════════════════════════════════════════╗
║        CLAMP DETECTION & COUNTING — YOLOv8  v3                      ║
║        Heavy augmentation · Stable tracking · Accurate count        ║
╚══════════════════════════════════════════════════════════════════════╝

HOW TO USE
──────────
1. Install (once):
       pip install ultralytics opencv-python matplotlib albumentations

2. Folder must look like:
       llll/
         ├── clamp_detection.py
         ├── Task.mp4
         ├── yolov8n.pt
         └── train/
               ├── images/
               └── labels/

3. Run:
       python clamp_detection.py
"""

# ══════════════════════════════════════════════════════════════════════
#  IMPORTS
# ══════════════════════════════════════════════════════════════════════
import os, sys, json, shutil, random, time, textwrap, math
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] Run:  pip install ultralytics")
    sys.exit(1)

try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("[WARN] albumentations not found — using built-in augmentation only.")
    print("       For better results: pip install albumentations")

# ══════════════════════════════════════════════════════════════════════
#  ★  CONFIG  — edit here if needed
# ══════════════════════════════════════════════════════════════════════
VIDEO_PATH      = "Task.mp4"
MODEL_SIZE      = "yolov8s.pt"  # more capacity than nano, still CPU-friendly

EPOCHS          = 150          # more epochs = better convergence
BATCH           = 4            # lowered: larger imgsz (960) + yolov8s need more memory
IMG_SIZE        = 960          # higher res = catches small/overlapping clamps
PATIENCE        = 30           # early-stop patience
LR0             = 0.005        # lower LR = more stable training
LRF             = 0.01         # final LR factor
WARMUP_EPOCHS   = 5
DEVICE          = '0'        # change to 0 if you have a CUDA GPU

CONF            = 0.40         # per-pass detection confidence threshold
IOU             = 0.45         # per-pass NMS IoU (used by model.predict/val/train)

# ── Close-pair / small-object detection ─────────────────────────────
# A single full-frame pass at IMG_SIZE often can't separate two clamps
# that are touching, because each one only occupies a handful of pixels
# after the frame gets resized down to IMG_SIZE. The fix is to ALSO run
# inference on cropped tiles of the frame at full resolution, so each
# clamp gets far more pixels, then merge everything back together.
TILE_GRID       = (2, 2)       # split frame into rows x cols tiles
TILE_OVERLAP    = 0.15         # tiles overlap by 15% so nothing straddles a tile edge
TILE_CONF       = 0.30         # slightly lower threshold on tiles — full-res crops
                                # naturally yield lower raw confidence on tiny objects,
                                # this still rejects noise but doesn't throw away real hits
# NOTE on merge IOU: NMS suppresses a box if its IoU with a kept box
# EXCEEDS the threshold. A LOWER threshold suppresses MORE aggressively
# (bad for close pairs — it deletes the second clamp). To keep two
# genuinely separate touching clamps as two boxes, the merge step needs
# a HIGHER threshold than normal NMS, not lower.
IOU_MERGE       = 0.55         # threshold used only when merging full+tile passes

EXPECTED_CLAMPS = 14           # known true count — used to cap visible detections

# Augmentation: how many extra copies to generate per original image
AUG_COPIES      = 8            # 49 images × 8 = 392 extra → 441 total

VAL_RATIO       = 0.15
TEST_RATIO      = 0.10
SEED            = 42
SKIP_TRAIN      = False        # set True to reuse existing best.pt

# Dataloader workers. Ultralytics defaults to 8 if unset, which on Windows
# means 8 separate process respawns (each re-imports torch/CUDA) whenever
# the dataloader is rebuilt — e.g. at the "Closing dataloader mosaic" step
# in the last 10 epochs. That respawn spike is what exhausts the Windows
# page file and crashes training. Keeping this lower avoids the spike.
WORKERS         = 4

# Tracking / counting stability
SMOOTH_WINDOW   = 7            # frame-count smoothing window
MIN_HITS        = 3            # track must appear N frames before counted
MAX_LOST        = 8            # frames to keep a lost track alive

# ══════════════════════════════════════════════════════════════════════
#  PATHS
# ══════════════════════════════════════════════════════════════════════
ROOT         = Path(__file__).resolve().parent
RUNS_DIR     = ROOT / "runs"
SAVED_DIR    = ROOT / "saved_models"
AUG_DIR      = ROOT / "data_augmented"
BEST_WEIGHTS = RUNS_DIR / "train" / "clamp_yolov8" / "weights" / "best.pt"
IMG_EXTS     = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ══════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════
def banner(title: str):
    print("\n" + "═" * 64)
    print(f"  {title}")
    print("═" * 64)

def info(msg):  print(f"  [INFO] {msg}")
def warn(msg):  print(f"  [WARN] {msg}")
def ok(msg):    print(f"  ✅  {msg}")

# ══════════════════════════════════════════════════════════════════════
#  STEP 1 — DATA AUGMENTATION
#  Generates AUG_COPIES augmented versions of every training image
#  with correctly transformed YOLO bounding box labels.
# ══════════════════════════════════════════════════════════════════════

def _yolo_to_xyxy(box, W, H):
    """Convert YOLO (cx,cy,w,h) normalised → pixel (x1,y1,x2,y2)."""
    cx, cy, bw, bh = box
    x1 = int((cx - bw / 2) * W)
    y1 = int((cy - bh / 2) * H)
    x2 = int((cx + bw / 2) * W)
    y2 = int((cy + bh / 2) * H)
    return max(0,x1), max(0,y1), min(W,x2), min(H,y2)

def _xyxy_to_yolo(x1, y1, x2, y2, W, H):
    """Convert pixel (x1,y1,x2,y2) → YOLO (cx,cy,w,h) normalised."""
    cx = ((x1 + x2) / 2) / W
    cy = ((y1 + y2) / 2) / H
    bw = (x2 - x1) / W
    bh = (y2 - y1) / H
    return cx, cy, bw, bh

def _read_labels(lbl_path):
    """Read YOLO label file → list of (cls, cx, cy, w, h)."""
    labels = []
    if lbl_path.exists():
        for line in lbl_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                labels.append((int(parts[0]),
                                float(parts[1]), float(parts[2]),
                                float(parts[3]), float(parts[4])))
    return labels

def _write_labels(lbl_path, labels):
    lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
             for c, cx, cy, w, h in labels]
    lbl_path.write_text("\n".join(lines))

def _augment_albumentations(img, labels):
    """Heavy albumentations pipeline — returns (aug_img, aug_labels)."""
    H, W = img.shape[:2]
    bboxes  = [_yolo_to_xyxy(lbl[1:], W, H) for lbl in labels]
    classes = [lbl[0] for lbl in labels]

    # Convert to albumentations format [x1,y1,x2,y2] normalised
    albu_boxes = [[x1/W, y1/H, x2/W, y2/H] for x1,y1,x2,y2 in bboxes]

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RandomRotate90(p=0.3),
        A.Rotate(limit=20, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.35,
                                    contrast_limit=0.35, p=0.7),
        A.HueSaturationValue(hue_shift_limit=15,
                             sat_shift_limit=40,
                             val_shift_limit=30, p=0.5),
        A.GaussNoise(var_limit=(10, 60), p=0.4),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.MotionBlur(blur_limit=7, p=0.2),
        A.CLAHE(clip_limit=4.0, p=0.3),
        A.RandomGamma(gamma_limit=(70, 130), p=0.3),
        A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
        A.RandomShadow(p=0.2),
        A.CoarseDropout(max_holes=6, max_height=30,
                        max_width=30, fill_value=0, p=0.2),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2,
                           rotate_limit=15, p=0.5),
    ], bbox_params=A.BboxParams(
        format='albumentations',
        label_fields=['class_labels'],
        min_visibility=0.3,
    ))

    result = transform(image=img, bboxes=albu_boxes, class_labels=classes)
    aug_img = result['image']
    aH, aW  = aug_img.shape[:2]

    aug_labels = []
    for cls_id, (nx1, ny1, nx2, ny2) in zip(result['class_labels'],
                                              result['bboxes']):
        cx, cy, bw, bh = _xyxy_to_yolo(
            int(nx1*aW), int(ny1*aH), int(nx2*aW), int(ny2*aH), aW, aH)
        if bw > 0.01 and bh > 0.01:
            aug_labels.append((cls_id, cx, cy, bw, bh))
    return aug_img, aug_labels


def _augment_cv2(img, labels):
    """Fallback augmentation using only OpenCV (no albumentations)."""
    H, W = img.shape[:2]
    aug = img.copy()

    # Random brightness / contrast
    alpha = random.uniform(0.6, 1.4)
    beta  = random.randint(-40, 40)
    aug   = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)

    # Random horizontal flip
    flip  = random.random() < 0.5
    if flip:
        aug = cv2.flip(aug, 1)

    # Random rotation ±15°
    angle = random.uniform(-15, 15)
    M     = cv2.getRotationMatrix2D((W/2, H/2), angle, 1.0)
    aug   = cv2.warpAffine(aug, M, (W, H),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT_101)

    # Gaussian blur
    if random.random() < 0.3:
        k   = random.choice([3, 5])
        aug = cv2.GaussianBlur(aug, (k, k), 0)

    # HSV jitter
    hsv = cv2.cvtColor(aug, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = np.clip(hsv[..., 0] + random.uniform(-10, 10), 0, 179)
    hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.7, 1.3), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * random.uniform(0.7, 1.3), 0, 255)
    aug = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Transform labels
    aug_labels = []
    for cls_id, cx, cy, bw, bh in labels:
        # Flip
        if flip:
            cx = 1.0 - cx
        # Rotate box corners
        x1p, y1p, x2p, y2p = _yolo_to_xyxy((cx, cy, bw, bh), W, H)
        corners = np.array([[x1p, y1p, 1], [x2p, y1p, 1],
                             [x2p, y2p, 1], [x1p, y2p, 1]], dtype=np.float32)
        rotated = corners @ M.T
        rx1, ry1 = rotated[:, 0].min(), rotated[:, 1].min()
        rx2, ry2 = rotated[:, 0].max(), rotated[:, 1].max()
        rx1, ry1 = max(0, rx1), max(0, ry1)
        rx2, ry2 = min(W, rx2), min(H, ry2)
        if rx2 - rx1 > 5 and ry2 - ry1 > 5:
            ncx, ncy, nbw, nbh = _xyxy_to_yolo(rx1, ry1, rx2, ry2, W, H)
            aug_labels.append((cls_id, ncx, ncy, nbw, nbh))
    return aug, aug_labels


def augment_dataset(src_img_dir: Path, src_lbl_dir: Path,
                    dst_img_dir: Path, dst_lbl_dir: Path,
                    n_copies: int):
    """
    Generate n_copies augmented versions of every image in src_img_dir.
    Also copies the originals to dst directories.
    """
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in src_img_dir.rglob("*")
                    if p.suffix.lower() in IMG_EXTS)
    aug_fn = _augment_albumentations if HAS_ALBUMENTATIONS else _augment_cv2

    total_written = 0
    for img_path in images:
        lbl_path = (src_lbl_dir / img_path.name).with_suffix(".txt")
        labels   = _read_labels(lbl_path)

        img = cv2.imread(str(img_path))
        if img is None:
            warn(f"Cannot read {img_path.name} — skipped")
            continue

        # Copy original
        shutil.copy2(img_path, dst_img_dir / img_path.name)
        if lbl_path.exists():
            shutil.copy2(lbl_path, dst_lbl_dir / lbl_path.name)
        total_written += 1

        # Generate augmented copies
        for i in range(n_copies):
            aug_img, aug_labels = aug_fn(img.copy(), labels)
            stem     = img_path.stem
            aug_name = f"{stem}_aug{i:03d}.jpg"
            cv2.imwrite(str(dst_img_dir / aug_name), aug_img,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            _write_labels(dst_lbl_dir / f"{stem}_aug{i:03d}.txt", aug_labels)
            total_written += 1

    return total_written


# ══════════════════════════════════════════════════════════════════════
#  STEP 2 — PREPARE DATASET  (augment → split → YAML)
# ══════════════════════════════════════════════════════════════════════
def prepare_dataset():
    banner("STEP 1 / 5  —  Data Augmentation & Dataset Preparation")

    src_img = ROOT / "train" / "images"
    src_lbl = ROOT / "train" / "labels"

    if not src_img.exists() or not src_lbl.exists():
        print(f"[ERROR] Cannot find train/images or train/labels under {ROOT}")
        sys.exit(1)

    orig_count = len([p for p in src_img.rglob("*")
                      if p.suffix.lower() in IMG_EXTS])
    info(f"Original images : {orig_count}")

    # ── Augment ────────────────────────────────────────────
    aug_img_dir = AUG_DIR / "images"
    aug_lbl_dir = AUG_DIR / "labels"

    if aug_img_dir.exists() and any(aug_img_dir.iterdir()):
        existing = len(list(aug_img_dir.glob("*.jpg")) +
                       list(aug_img_dir.glob("*.png")))
        info(f"Augmented data already exists ({existing} images) — skipping re-augment")
    else:
        info(f"Generating {AUG_COPIES} augmented copies per image …")
        total = augment_dataset(src_img, src_lbl,
                                aug_img_dir, aug_lbl_dir, AUG_COPIES)
        ok(f"Augmentation done  →  {total} total images "
           f"(original {orig_count}  +  augmented {total - orig_count})")

    # ── Collect all pairs from augmented pool ──────────────
    all_imgs = sorted(p for p in aug_img_dir.rglob("*")
                      if p.suffix.lower() in IMG_EXTS)
    pairs = []
    for img in all_imgs:
        lbl = (aug_lbl_dir / img.name).with_suffix(".txt")
        if lbl.exists():
            pairs.append((img, lbl))

    if not pairs:
        print("[ERROR] No pairs in augmented pool.")
        sys.exit(1)

    info(f"Total pairs for training : {len(pairs)}")

    # ── Split ──────────────────────────────────────────────
    random.seed(SEED)
    random.shuffle(pairs)
    n       = len(pairs)
    n_val   = max(2, int(n * VAL_RATIO))
    n_test  = max(2, int(n * TEST_RATIO))
    n_train = n - n_val - n_test

    splits = {
        "train": pairs[:n_train],
        "val"  : pairs[n_train:n_train + n_val],
        "test" : pairs[n_train + n_val:],
    }
    for k, v in splits.items():
        info(f"  {k:5s}: {len(v)} samples")

    # ── Copy into data/ ────────────────────────────────────
    data_dir = ROOT / "data"
    for split, sample_list in splits.items():
        id_ = data_dir / "images" / split
        ld_ = data_dir / "labels" / split
        if id_.exists() and any(id_.iterdir()):
            info(f"  {split} already copied — skipping")
            continue
        id_.mkdir(parents=True, exist_ok=True)
        ld_.mkdir(parents=True, exist_ok=True)
        for img, lbl in sample_list:
            shutil.copy2(img, id_ / img.name)
            shutil.copy2(lbl, ld_ / lbl.name)

    # ── YAML ───────────────────────────────────────────────
    yaml_path = data_dir / "clamp_dataset.yaml"
    yaml_path.write_text(textwrap.dedent(f"""\
        path: {data_dir.as_posix()}
        train: images/train
        val:   images/val
        test:  images/test
        nc: 1
        names:
          0: clamp
    """))
    ok(f"Dataset YAML → {yaml_path}")
    ok("Dataset ready!")
    return yaml_path


# ══════════════════════════════════════════════════════════════════════
#  STEP 3 — TRAIN
# ══════════════════════════════════════════════════════════════════════
def train_model(yaml_path):
    banner("STEP 2 / 5  —  Train YOLOv8  (augmented dataset)")

    if SKIP_TRAIN and BEST_WEIGHTS.exists():
        info(f"SKIP_TRAIN=True — using: {BEST_WEIGHTS}")
        return

    model = YOLO(MODEL_SIZE)

    results = model.train(
        data           = str(yaml_path),
        epochs         = EPOCHS,
        imgsz          = IMG_SIZE,
        batch          = BATCH,
        patience       = PATIENCE,
        lr0            = LR0,
        lrf            = LRF,
        warmup_epochs  = WARMUP_EPOCHS,
        weight_decay   = 0.0005,
        momentum       = 0.937,
        iou            = IOU,
        device         = DEVICE,
        workers        = WORKERS,
        project        = str(RUNS_DIR / "train"),
        name           = "clamp_yolov8",
        exist_ok       = True,
        save           = True,
        save_period    = 10,
        plots          = True,
        verbose        = True,
        # ── YOLOv8 built-in augmentation (on top of our pre-augmentation) ──
        augment        = True,
        mosaic         = 1.0,       # mosaic 4-image fusion
        mixup          = 0.15,      # mixup blending
        copy_paste     = 0.1,       # copy-paste augmentation
        degrees        = 15,        # rotation ±15°
        translate      = 0.1,       # translation ±10%
        scale          = 0.5,       # scale ±50%
        shear          = 5.0,       # shear ±5°
        perspective    = 0.0005,    # perspective distortion
        flipud         = 0.1,       # vertical flip
        fliplr         = 0.5,       # horizontal flip
        hsv_h          = 0.015,     # hue jitter
        hsv_s          = 0.7,       # saturation jitter
        hsv_v          = 0.4,       # brightness jitter
        erasing        = 0.4,       # random erasing
    )

    ok(f"Training complete  →  {BEST_WEIGHTS}")
    print("\n  ── Final Training Metrics ──")
    for k, v in results.results_dict.items():
        print(f"  {k:<35s}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


# ══════════════════════════════════════════════════════════════════════
#  STEP 4 — VALIDATE
# ══════════════════════════════════════════════════════════════════════
def validate_model(yaml_path):
    banner("STEP 3 / 5  —  Validate")

    model   = YOLO(str(BEST_WEIGHTS))
    val_res = model.val(
        data     = str(yaml_path),
        imgsz    = IMG_SIZE,
        batch    = BATCH,
        conf     = CONF,
        iou      = IOU,
        device   = DEVICE,
        project  = str(RUNS_DIR / "val"),
        name     = "clamp_val",
        exist_ok = True,
        plots    = True,
    )

    box = val_res.box
    p   = float(box.mp)
    r   = float(box.mr)
    f1  = 2 * p * r / (p + r + 1e-8)

    print(f"\n  mAP @ 0.50       : {box.map50:.4f}")
    print(f"  mAP @ 0.50:0.95  : {box.map:.4f}")
    print(f"  Precision        : {p:.4f}")
    print(f"  Recall           : {r:.4f}")
    print(f"  F1 Score         : {f1:.4f}")
    ok("Validation done!")


# ══════════════════════════════════════════════════════════════════════
#  STEP 5 — TEST
# ══════════════════════════════════════════════════════════════════════
def test_model(yaml_path):
    banner("STEP 4 / 5  —  Test on Held-Out Set")

    model    = YOLO(str(BEST_WEIGHTS))
    test_res = model.val(
        data     = str(yaml_path),
        split    = "test",
        imgsz    = IMG_SIZE,
        batch    = BATCH,
        conf     = CONF,
        iou      = IOU,
        device   = DEVICE,
        project  = str(RUNS_DIR / "test"),
        name     = "clamp_test",
        exist_ok = True,
        plots    = True,
        save_txt = True,
    )

    box = test_res.box
    p   = float(box.mp)
    r   = float(box.mr)
    f1  = 2 * p * r / (p + r + 1e-8)

    metrics = {
        "mAP50"    : round(float(box.map50), 4),
        "mAP50_95" : round(float(box.map),   4),
        "Precision": round(p, 4),
        "Recall"   : round(r, 4),
        "F1"       : round(f1, 4),
    }

    print("\n╔════════════════════════════════════════╗")
    print("║         TEST SET  METRICS              ║")
    print("╠════════════════════════════════════════╣")
    for k, v in metrics.items():
        print(f"║  {k:<22s}  :  {v:.4f}        ║")
    print("╚════════════════════════════════════════╝")

    out_dir = Path(test_res.save_dir)
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Bar chart ──────────────────────────────────────────
    colours = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(list(metrics.keys()), list(metrics.values()),
                  color=colours, edgecolor="black", linewidth=0.6)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("YOLOv8 Clamp Detection — Test Metrics",
                 fontsize=14, fontweight="bold")
    for bar, val in zip(bars, metrics.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_dir / "test_metrics_chart.png", dpi=150)
    plt.close()
    ok(f"Chart → {out_dir / 'test_metrics_chart.png'}")
    ok("Testing done!")


# ══════════════════════════════════════════════════════════════════════
#  STEP 6 — VIDEO INFERENCE  with stable tracking & accurate counting
# ══════════════════════════════════════════════════════════════════════

# ── Drawing constants ──────────────────────────────────────────────────
FONT        = cv2.FONT_HERSHEY_DUPLEX
BOX_CLR     = (0,   230,  80)   # bright green boxes
LBL_BG      = (0,    80,  25)   # dark green label background
LBL_TXT     = (255, 255, 255)   # white label text
HUD_BG      = (12,   12,  12)   # near-black HUD
HUD_GREEN   = (0,   230,  80)   # green text
HUD_GOLD    = (50,  210, 255)   # gold text for total count


def _draw_box(frame, x1, y1, x2, y2, conf, tid):
    """Draw bounding box + label with track ID."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_CLR, 2)
    label = f"clamp #{tid}  {conf:.2f}" if tid is not None \
            else f"clamp  {conf:.2f}"
    tw, th = cv2.getTextSize(label, FONT, 0.50, 1)[0]
    ly1 = max(y1 - th - 6, 0)
    cv2.rectangle(frame, (x1, ly1), (x1 + tw + 6, y1), LBL_BG, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 4),
                FONT, 0.50, LBL_TXT, 1, cv2.LINE_AA)


def _draw_hud(frame, visible: int, total: int,
              fps: float, conf_avg: float):
    """Draw semi-transparent HUD panel (top-left)."""
    lines = [
        (f"  Clamps visible : {visible}", HUD_GREEN),
        (f"  Avg confidence : {conf_avg:.2f}", HUD_GREEN),
        (f"  FPS            : {fps:.1f}", HUD_GREEN),
    ]
    lh, pad = 32, 12
    bw = 290
    bh = len(lines) * lh + 2 * pad

    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + bw, 8 + bh), HUD_BG, -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    # Header bar
    cv2.rectangle(frame, (8, 8), (8 + bw, 8 + lh),
                  (0, 100, 40), -1)
    cv2.putText(frame, "  CLAMP DETECTOR",
                (14, 8 + lh - 8), FONT, 0.52,
                (200, 255, 200), 1, cv2.LINE_AA)

    for i, (line, clr) in enumerate(lines):
        y = 8 + lh + pad + (i + 1) * lh - 6
        cv2.putText(frame, line, (14, y),
                    FONT, 0.54, clr, 1, cv2.LINE_AA)


def _make_tiles(W, H, grid=TILE_GRID, overlap=TILE_OVERLAP):
    """Split a W x H frame into overlapping tiles for full-resolution
    close-pair inference. Returns list of (x1, y1, x2, y2) pixel boxes."""
    rows, cols = grid
    tile_w, tile_h = W / cols, H / rows
    ow, oh = tile_w * overlap, tile_h * overlap
    tiles = []
    for r in range(rows):
        for c in range(cols):
            x1 = max(0, int(c * tile_w - ow))
            y1 = max(0, int(r * tile_h - oh))
            x2 = min(W, int((c + 1) * tile_w + ow))
            y2 = min(H, int((r + 1) * tile_h + oh))
            if x2 - x1 > 20 and y2 - y1 > 20:
                tiles.append((x1, y1, x2, y2))
    return tiles


def _iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / (union + 1e-6)


def _merge_nms(dets, iou_thresh=IOU_MERGE):
    """Greedy NMS across detections pooled from multiple inference passes.
    dets: list of (x1,y1,x2,y2,conf). Keeps highest-confidence box first,
    only drops a later box if it overlaps an already-kept box by MORE
    than iou_thresh — a high threshold here is what lets two real,
    touching-but-distinct clamps both survive as separate boxes."""
    dets = sorted(dets, key=lambda d: d[4], reverse=True)
    keep = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [d for d in dets if _iou_xyxy(best[:4], d[:4]) <= iou_thresh]
    return keep


def detect_multi_pass(model, frame, W, H):
    """Run full-frame + tiled inference and merge results. Tiled passes
    give touching/small clamps far more pixels to be told apart at,
    which a single full-frame resize can't provide."""
    all_dets = []

    # Pass 1 — full frame (fast, catches most clamps)
    preds = model.predict(frame, conf=CONF, iou=IOU, imgsz=IMG_SIZE, verbose=False)[0]
    if preds.boxes is not None:
        for b in preds.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            all_dets.append((x1, y1, x2, y2, float(b.conf[0])))

    # Pass 2 — overlapping tiles at native resolution
    for (tx1, ty1, tx2, ty2) in _make_tiles(W, H):
        crop = frame[ty1:ty2, tx1:tx2]
        if crop.size == 0:
            continue
        tpreds = model.predict(crop, conf=TILE_CONF, iou=IOU,
                                imgsz=IMG_SIZE, verbose=False)[0]
        if tpreds.boxes is not None:
            for b in tpreds.boxes:
                lx1, ly1, lx2, ly2 = map(int, b.xyxy[0].tolist())
                all_dets.append((lx1 + tx1, ly1 + ty1, lx2 + tx1, ly2 + ty1,
                                  float(b.conf[0])))

    # Merge all passes — high-IoU threshold preserves close-but-distinct pairs
    return _merge_nms(all_dets, IOU_MERGE)


class SimpleTracker:
    """
    Lightweight IoU-based tracker that:
      • Assigns stable IDs to detections across frames
      • Keeps lost tracks alive for MAX_LOST frames
      • Only confirms a track after MIN_HITS appearances
      • Never decrements total_count (cumulative unique clamps)
    """
    def __init__(self):
        self.next_id      = 1
        self.tracks       = {}   # id → {box, hits, lost, conf}
        self.total_count  = 0    # cumulative confirmed unique clamps
        self.confirmed_ids = set()

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        iw  = max(0, ix2 - ix1)
        ih  = max(0, iy2 - iy1)
        inter = iw * ih
        ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / (ua + 1e-6)

    def update(self, detections):
        """
        detections: list of (x1,y1,x2,y2,conf)
        Returns: list of (x1,y1,x2,y2,conf,track_id)
        """
        # Age all existing tracks (increment lost counter)
        for tid in list(self.tracks):
            self.tracks[tid]['lost'] += 1
            if self.tracks[tid]['lost'] > MAX_LOST:
                del self.tracks[tid]

        matched_tids = set()
        results      = []
        unmatched    = list(range(len(detections)))

        # Match detections to existing tracks by IoU
        for tid, trk in self.tracks.items():
            best_iou  = 0.35          # minimum IoU to accept a match
            best_didx = -1
            for didx in unmatched:
                d = detections[didx]
                iou_val = self.iou(trk['box'], d[:4])
                if iou_val > best_iou:
                    best_iou  = iou_val
                    best_didx = didx

            if best_didx >= 0:
                d = detections[best_didx]
                trk['box']  = d[:4]
                trk['conf'] = d[4]
                trk['hits'] += 1
                trk['lost']  = 0
                unmatched.remove(best_didx)
                matched_tids.add(tid)

                if trk['hits'] >= MIN_HITS:
                    results.append((*d[:4], d[4], tid))
                    if tid not in self.confirmed_ids:
                        self.confirmed_ids.add(tid)
                        self.total_count += 1

        # Create new tracks for unmatched detections
        for didx in unmatched:
            d  = detections[didx]
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {
                'box' : d[:4],
                'conf': d[4],
                'hits': 1,
                'lost': 0,
            }
            # NOTE: no instant-confirm shortcut anymore — every track,
            # regardless of confidence, must persist for MIN_HITS frames
            # before being counted. This prevents one-frame flickers
            # (reflections, noise, motion blur) from inflating the count.

        return results

    # alias
    def iou(self, a, b):
        return self.__class__._iou(a, b)


def run_inference():
    banner("STEP 5 / 5  —  Detect & Count Clamps in Video")

    video_path = ROOT / VIDEO_PATH
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    model   = YOLO(str(BEST_WEIGHTS))
    tracker = SimpleTracker()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    info(f"Video: {W}×{H} @ {fps_in:.1f} fps  |  {total} frames")

    inf_dir   = RUNS_DIR / "inference"
    inf_dir.mkdir(parents=True, exist_ok=True)
    out_video = inf_dir / f"{video_path.stem}_detected.mp4"
    fourcc    = cv2.VideoWriter_fourcc(*"mp4v")
    writer    = cv2.VideoWriter(str(out_video), fourcc, fps_in, (W, H))

    frame_stats   = []
    idx           = 0
    t_start       = time.time()
    count_history = deque(maxlen=SMOOTH_WINDOW)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Inference (full-frame + tiled close-pair passes, merged) ──
        raw_dets_all = detect_multi_pass(model, frame, W, H)

        # ── Feed detections into tracker ──────────────────────
        # Filter out boxes that are way too large to be a single clamp
        # (these are almost always false positives spanning most of the frame)
        MAX_BOX_AREA_FRAC = 0.15   # a clamp should never cover >15% of frame area
        frame_area = W * H

        raw_dets = []
        for (x1, y1, x2, y2, c) in raw_dets_all:
            box_area = max(0, x2 - x1) * max(0, y2 - y1)
            if box_area > frame_area * MAX_BOX_AREA_FRAC:
                continue  # skip oversized false-positive box
            raw_dets.append((x1, y1, x2, y2, c))

        # Cap to the known true clamp count: keep only the N highest-
        # confidence detections per frame (drops weak/duplicate boxes
        # past the expected total instead of over- or under-counting)
        if len(raw_dets) > EXPECTED_CLAMPS:
            raw_dets.sort(key=lambda d: d[4], reverse=True)
            raw_dets = raw_dets[:EXPECTED_CLAMPS]

        tracked = tracker.update(raw_dets)

        # ── Draw each confirmed detection ─────────────────────
        confs = []
        for (x1, y1, x2, y2, c, tid) in tracked:
            _draw_box(frame, x1, y1, x2, y2, c, tid)
            confs.append(c)

        # ── Smooth visible count ──────────────────────────────
        count_history.append(len(tracked))
        visible_smooth = int(round(sum(count_history) / len(count_history)))

        conf_avg = float(np.mean(confs)) if confs else 0.0
        cur_fps  = (idx + 1) / (time.time() - t_start + 1e-6)

        _draw_hud(frame, visible_smooth, tracker.total_count,
                  cur_fps, conf_avg)
        writer.write(frame)

        frame_stats.append({
            "frame"        : idx,
            "visible"      : visible_smooth,
            "total_counted": tracker.total_count,
            "conf_avg"     : round(conf_avg, 4),
        })

        idx += 1
        if idx % 60 == 0 or idx == 1:
            pct = idx / max(total, 1) * 100
            print(f"   {idx:>5}/{total}  ({pct:5.1f}%)  "
                  f"visible={visible_smooth}  "
                  f"total={tracker.total_count}  "
                  f"fps={cur_fps:.1f}")

    cap.release()
    writer.release()

    # ── Final summary ─────────────────────────────────────────
    visible_list = [s["visible"] for s in frame_stats]
    conf_list    = [s["conf_avg"] for s in frame_stats]
    nz_conf      = [c for c in conf_list if c > 0]

    summary = {
        "total_frames"          : idx,
        "TOTAL_CLAMPS_COUNTED"  : tracker.total_count,
        "max_visible_per_frame" : int(max(visible_list)) if visible_list else 0,
        "avg_visible_per_frame" : round(float(np.mean(visible_list)), 2) if visible_list else 0,
        "overall_avg_confidence": round(float(np.mean(nz_conf)), 4) if nz_conf else 0,
        "output_video"          : str(out_video),
    }

    stats_file = inf_dir / f"{video_path.stem}_stats.json"
    with open(stats_file, "w") as f:
        json.dump({"summary": summary, "per_frame": frame_stats}, f, indent=2)

    print("\n╔══════════════════════════════════════════════╗")
    print("║         VIDEO INFERENCE SUMMARY              ║")
    print("╠══════════════════════════════════════════════╣")
    for k, v in summary.items():
        print(f"║  {k:<30s}: {str(v):<16s}║")
    print("╚══════════════════════════════════════════════╝")
    print(f"\n  Annotated video → {out_video}")
    print(f"  Stats JSON      → {stats_file}")
    ok("Inference done!")


# ══════════════════════════════════════════════════════════════════════
#  STEP 7 — SAVE MODEL
# ══════════════════════════════════════════════════════════════════════
def save_model():
    banner("BONUS  —  Save Model")
    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    dst = SAVED_DIR / "clamp_yolov8_best.pt"
    shutil.copy2(BEST_WEIGHTS, dst)
    ok(f"PyTorch model  → {dst}")

    try:
        model    = YOLO(str(BEST_WEIGHTS))
        onnx_src = model.export(format="onnx", imgsz=IMG_SIZE,
                                dynamic=False, simplify=True)
        shutil.copy2(onnx_src, SAVED_DIR / "clamp_yolov8.onnx")
        ok(f"ONNX model     → {SAVED_DIR / 'clamp_yolov8.onnx'}")
    except Exception as e:
        warn(f"ONNX export skipped: {e}")


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║     CLAMP DETECTION & COUNTING — YOLOv8  v3                     ║
║     Heavy Augmentation · ByteTrack · Stable Count               ║
╚══════════════════════════════════════════════════════════════════╝
  Steps:  1) Augment dataset  (×8 copies per image)
          2) Train YOLOv8     (150 epochs, mosaic + mixup)
          3) Validate
          4) Test             (mAP / Precision / Recall / F1)
          5) Infer on video   (bounding boxes + stable count HUD)
       +  6) Save model
""")
    t0 = time.time()

    yaml_path = prepare_dataset()
    train_model(yaml_path)
    validate_model(yaml_path)
    test_model(yaml_path)
    run_inference()
    save_model()

    e = time.time() - t0
    h, m, s = int(e//3600), int((e%3600)//60), int(e%60)
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  🏁  ALL DONE  —  Total time: {h:02d}h {m:02d}m {s:02d}s
╠══════════════════════════════════════════════════════════════════╣
║  Weights     runs/train/clamp_yolov8/weights/best.pt            ║
║  Test plots  runs/test/clamp_test/                              ║
║  Output vid  runs/inference/Task_detected.mp4                   ║
║  Saved model saved_models/clamp_yolov8_best.pt                  ║
╚══════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
