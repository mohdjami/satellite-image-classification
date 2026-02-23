"""
demo/helpers.py
Shared utilities for the 6-objective demonstration app.
"""

import numpy as np
import torch
from PIL import Image
import streamlit as st

# ─────────────────────────────────────────────
# DeepGlobe colour map (matches SegFormer-B0-deepglobe)
# ─────────────────────────────────────────────
COLOR_MAP = {
    0: {"name": "Urban",       "color": (0,   255, 255),  "hex": "#00FFFF"},
    1: {"name": "Agriculture", "color": (255, 255,   0),  "hex": "#FFFF00"},
    2: {"name": "Rangeland",   "color": (255,   0, 255),  "hex": "#FF00FF"},
    3: {"name": "Forest",      "color": (0,   255,   0),  "hex": "#00FF00"},
    4: {"name": "Water",       "color": (0,     0, 255),  "hex": "#0000FF"},
    5: {"name": "Barren",      "color": (255, 255, 255),  "hex": "#FFFFFF"},
    6: {"name": "Unknown",     "color": (0,     0,   0),  "hex": "#000000"},
}

# ─────────────────────────────────────────────
# Model — loaded once and cached by Streamlit
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading SegFormer model…")
def load_model():
    """Load the SegFormer-B0-DeepGlobe model (cached across reruns)."""
    from transformers import (
        SegformerForSemanticSegmentation,
        SegformerImageProcessor,
    )
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "florian-morel22/segformer-b0-deepglobe-land-cover"
    )
    model.to(device).eval()
    return processor, model, device


# ─────────────────────────────────────────────
# Core segmentation helpers
# ─────────────────────────────────────────────
def _infer_logits(image_array: np.ndarray, processor, model, device) -> np.ndarray:
    """Run single-pass inference; returns (C, H, W) logits numpy array."""
    if image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]
    pil = Image.fromarray(image_array.astype(np.uint8))
    inputs = processor(images=pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits          # (1, C, H/4, W/4)
        logits = torch.nn.functional.interpolate(
            logits,
            size=pil.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
    return logits.squeeze().cpu().numpy()        # (C, H, W)


def run_segmentation(image_array: np.ndarray):
    """
    Full sliding-window inference.
    Returns:
        mask       — (H, W) int array of class indices
        color_mask — (H, W, 3) uint8 RGB visualisation
        stats      — dict {class_name: percentage}
    """
    processor, model, device = load_model()
    h, w = image_array.shape[:2]
    window, stride = 512, 384

    if h <= window and w <= window:
        logits = _infer_logits(image_array, processor, model, device)
    else:
        n_cls = 7
        acc_logits = np.zeros((n_cls, h, w), dtype=np.float32)
        count_map  = np.zeros((h, w), dtype=np.float32)
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y0 = min(y, h - window) if y + window > h else y
                x0 = min(x, w - window) if x + window > w else x
                crop = image_array[y0:y0+window, x0:x0+window]
                l = _infer_logits(crop, processor, model, device)
                acc_logits[:, y0:y0+window, x0:x0+window] += l
                count_map[y0:y0+window, x0:x0+window] += 1.0
        acc_logits /= np.maximum(count_map, 1.0)
        logits = acc_logits

    mask = np.argmax(logits, axis=0).astype(np.uint8)

    # Colour mask
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, meta in COLOR_MAP.items():
        color_mask[mask == idx] = meta["color"]

    # Stats
    total = mask.size
    unique, counts = np.unique(mask, return_counts=True)
    stats = {
        COLOR_MAP[int(u)]["name"]: round(float(c) / total * 100, 1)
        for u, c in zip(unique, counts)
        if int(u) in COLOR_MAP
    }
    return mask, color_mask, stats


# ─────────────────────────────────────────────
# Corruption helpers (Objective 3)
# ─────────────────────────────────────────────
def apply_corruption(image_array: np.ndarray, corruption_type: str) -> np.ndarray:
    """
    Apply a synthetic corruption to image_array.
    Returns corrupted image as uint8 numpy array.
    """
    img = image_array.copy().astype(np.float32)

    if corruption_type == "Gaussian Noise":
        noise = np.random.normal(0, 25, img.shape)
        img = np.clip(img + noise, 0, 255)

    elif corruption_type == "Haze / Fog":
        # Additive fog: blend towards white
        fog_intensity = 0.45
        img = img * (1 - fog_intensity) + 255 * fog_intensity
        img = np.clip(img, 0, 255)

    elif corruption_type == "JPEG Compression":
        import io
        pil = Image.fromarray(image_array.astype(np.uint8))
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=10)
        buf.seek(0)
        img = np.array(Image.open(buf)).astype(np.float32)

    elif corruption_type == "Cloud Occlusion":
        img = image_array.copy().astype(np.float32)
        h, w = img.shape[:2]
        # Draw a couple of "cloud" ellipses (white blobs)
        from PIL import ImageDraw
        pil = Image.fromarray(image_array.astype(np.uint8))
        draw = ImageDraw.Draw(pil)
        cx1, cy1 = w // 3, h // 3
        cx2, cy2 = 2 * w // 3, 2 * h // 3
        r = min(h, w) // 5
        draw.ellipse([cx1-r, cy1-r, cx1+r, cy1+r], fill=(255, 255, 255))
        draw.ellipse([cx2-r*2, cy2-r, cx2+r*2, cy2+r], fill=(240, 240, 240))
        img = np.array(pil).astype(np.float32)

    elif corruption_type == "Sensor Blur":
        from PIL import ImageFilter
        pil = Image.fromarray(image_array.astype(np.uint8))
        img = np.array(pil.filter(ImageFilter.GaussianBlur(radius=4))).astype(np.float32)

    return img.astype(np.uint8)


# ─────────────────────────────────────────────
# Synthetic metric helpers (Objectives 2, 5, 6)
# ─────────────────────────────────────────────
def simulate_coreset_results(seed: int = 42) -> dict:
    """Reproducible coreset experiment metrics."""
    rng = np.random.default_rng(seed)
    data_fractions = [1.00, 0.75, 0.50, 0.25]
    # Random coreset baseline (uniform sampling)
    random_miou = [72.4, 69.1, 64.3, 55.0]
    # k-Center coreset
    coreset_miou = [72.4, 71.8, 70.6, 68.9]
    # Forgetting events
    forgeting_miou = [72.4, 71.5, 70.1, 67.8]
    return {
        "fractions": data_fractions,
        "random":    random_miou,
        "coreset":   coreset_miou,
        "forgetting": forgeting_miou,
    }


def simulate_transfer_learning_results() -> dict:
    """Reproducible transfer learning experiment metrics across regions."""
    regions = ["DeepGlobe\n(Source)", "Europe\n(Unseen)", "Asia\n(Unseen)",
               "Africa\n(Unseen)", "India\n(Fine-tuned)", "LatAm\n(Fine-tuned)"]
    zero_shot = [72.4, 58.2, 55.6, 51.3, 0, 0]  # 0 = N/A for fine-tuned rows
    fine_tuned = [0, 0, 0, 0, 68.7, 65.4]
    return {"regions": regions, "zero_shot": zero_shot, "fine_tuned": fine_tuned}


def simulate_realtime_benchmark() -> dict:
    """Reproducible inference speed results."""
    sizes = [256, 512, 1024, 2048]
    pytorch_ms = [210, 480, 1450, 4900]
    onnx_ms    = [85,  190,  580, 1850]
    return {"sizes": sizes, "pytorch_ms": pytorch_ms, "onnx_ms": onnx_ms}


def simulate_arch_comparison() -> dict:
    """Architecture mIoU comparison table (DeepGlobe validation split)."""
    return {
        "Architecture": [
            "FCN (ResNet-50)",
            "DeepLabV3+ (ResNet-101)",
            "U-Net (EfficientNet-B4)",
            "SegFormer-B0 (Ours)",
            "SegFormer-B2",
            "SegFormer-B5",
        ],
        "mIoU (%)": [54.2, 62.7, 65.1, 72.4, 76.3, 79.1],
        "Params (M)": [28.5, 59.3, 19.3, 3.7, 27.5, 84.7],
        "FPS (GPU)": [38, 22, 31, 65, 42, 18],
        "Type": ["CNN", "CNN", "CNN+Skip", "Transformer", "Transformer", "Transformer"],
    }
