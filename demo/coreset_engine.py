"""
demo/coreset_engine.py
Real coreset selection pipeline:
  1. Extract actual SegFormer encoder embeddings from satellite patches
  2. Run k-Center Greedy coreset selection on real features
  3. Run t-SNE on real features for visualization
"""

import numpy as np
import torch
from PIL import Image
from sklearn.manifold import TSNE
import streamlit as st


# ─── Real embedding extraction ────────────────────────────────────────────────

def extract_embeddings(patches: list[np.ndarray], progress_cb=None) -> np.ndarray:
    """
    Extract patch-level feature embeddings from the SegFormer encoder.

    Each patch (256×256×3) is passed through the full SegFormer backbone;
    we take the final hidden state (B, C, H', W') and GAP it to get one
    C-dim vector per patch.

    Args:
        patches: list of (H, W, 3) uint8 numpy arrays
        progress_cb: optional callable(int, int) for progress updates

    Returns:
        embeddings: (N, D) float32 numpy array
    """
    from demo.helpers import load_model

    processor, model, device = load_model()

    model.eval()
    embeddings = []

    for i, patch in enumerate(patches):
        if patch.shape[-1] == 4:
            patch = patch[:, :, :3]
        pil = Image.fromarray(patch.astype(np.uint8))
        inputs = processor(images=pil, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            # Get encoder hidden states (4 stages of MiT)
            outputs = model(**inputs, output_hidden_states=True)
            # Use the last hidden state (most semantic): shape (B, C, H', W')
            last_hidden = outputs.hidden_states[-1]          # (1, C, H', W')
            # Global average pool → (C,)
            feat = last_hidden.mean(dim=[2, 3]).squeeze(0)   # (C,)
            embeddings.append(feat.cpu().float().numpy())

        if progress_cb:
            progress_cb(i + 1, len(patches))

    return np.stack(embeddings, axis=0)   # (N, C)


# ─── k-Center Greedy ─────────────────────────────────────────────────────────

def k_center_greedy(embeddings: np.ndarray, n_select: int, seed: int = 0) -> np.ndarray:
    """
    k-Center Greedy coreset selection.

    Iteratively adds the point that is farthest from the current selected set.
    This maximises the minimum distance from any unselected point to the set,
    bounding the approximation error of the full dataset.

    Args:
        embeddings: (N, D) feature matrix
        n_select:   number of samples to select
        seed:       index of the initial point

    Returns:
        selected_idx: (n_select,) int array of selected indices
    """
    N = len(embeddings)
    n_select = min(n_select, N)

    # Normalise to unit sphere for cosine-like distance
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    feats = embeddings / norms

    selected = [seed]
    # Distance from each point to the nearest selected point
    min_dists = np.full(N, np.inf)

    for _ in range(n_select - 1):
        # Update min distances with the newly added point
        last = feats[selected[-1]]
        dists_to_last = np.linalg.norm(feats - last, axis=1)
        min_dists = np.minimum(min_dists, dists_to_last)
        # Select the point farthest from the current set
        min_dists[selected] = -np.inf   # exclude already-selected
        selected.append(int(np.argmax(min_dists)))

    return np.array(selected)


# ─── t-SNE ───────────────────────────────────────────────────────────────────

def run_tsne(embeddings: np.ndarray, perplexity: float = 8.0,
             n_components: int = 2, seed: int = 42) -> np.ndarray:
    """
    Run t-SNE dimensionality reduction on real embeddings.

    Perplexity is capped at roughly N/5 as recommended.
    Returns (N, 2) float32 array.
    """
    perplexity = min(perplexity, max(2, len(embeddings) // 4))
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=seed,
        max_iter=1000,
        learning_rate="auto",
        init="pca" if len(embeddings) > 4 else "random",
    )
    return tsne.fit_transform(embeddings).astype(np.float32)


# ─── Convenience wrapper ──────────────────────────────────────────────────────

def run_full_coreset_pipeline(patches: list[np.ndarray],
                              n_select: int | None = None,
                              progress_cb=None):
    """
    End-to-end: patches → embeddings → coreset selection → t-SNE.

    Args:
        patches:     list of (H, W, 3) uint8 arrays
        n_select:    number of samples to select (default: 25% of N)
        progress_cb: optional callable(done, total) for embedding progress

    Returns:
        dict with keys:
            embeddings    (N, D) real feature vectors
            tsne_2d       (N, 2) t-SNE coordinates
            selected_idx  (K,)   indices chosen by k-Center Greedy
            mask          (N,)   bool mask, True = selected
    """
    embeddings = extract_embeddings(patches, progress_cb=progress_cb)
    N = len(embeddings)
    if n_select is None:
        n_select = max(1, N // 4)

    selected_idx = k_center_greedy(embeddings, n_select)
    tsne_2d = run_tsne(embeddings)

    mask = np.zeros(N, dtype=bool)
    mask[selected_idx] = True

    return {
        "embeddings":   embeddings,
        "tsne_2d":      tsne_2d,
        "selected_idx": selected_idx,
        "mask":         mask,
    }
