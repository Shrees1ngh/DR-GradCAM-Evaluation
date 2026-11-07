"""Cross-dataset perceptual similarity and visual duplicate verification.

Computes perceptual hashes (pHash - DCT frequency based, and dHash - gradient based)
to audit visual independence and detect potential image duplicates or near-duplicates
between APTOS 2019 and IDRiD cohorts.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger("similarity")


def compute_dhash(image: np.ndarray, hash_size: int = 8) -> np.ndarray:
    """Compute difference hash (dHash) tracking horizontal intensity gradients.

    Args:
        image: Grayscale (H, W) or RGB (H, W, 3) image array.
        hash_size: Hash grid dimension (default 8 produces 64-bit hash).

    Returns:
        1D boolean array of length hash_size * hash_size.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    return diff.flatten().astype(np.uint8)


def compute_phash(
    image: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4
) -> np.ndarray:
    """Compute perceptual hash (pHash) based on Discrete Cosine Transform (DCT).

    Args:
        image: Grayscale (H, W) or RGB (H, W, 3) image array.
        hash_size: Target low-frequency grid (default 8 produces 64-bit hash).
        highfreq_factor: Scaling factor for intermediate DCT representation.

    Returns:
        1D boolean array of length hash_size * hash_size.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    img_size = hash_size * highfreq_factor
    resized = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))

    # Extract lowest frequency DCT components (top-left block)
    dct_low = dct[:hash_size, :hash_size]

    # Compare against the median (excluding the DC component at (0, 0))
    med = np.median(dct_low[1:, 1:]) if hash_size > 1 else dct_low[0, 0]
    return (dct_low > med).flatten().astype(np.uint8)


def hash_single_image(image_path: Path) -> Optional[Tuple[str, np.ndarray, np.ndarray]]:
    """Load image from disk directly in grayscale and compute both dHash and pHash."""
    try:
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return None
        dh = compute_dhash(gray)
        ph = compute_phash(gray)
        return str(image_path), dh, ph
    except Exception as e:
        logger.warning(f"Error computing hash for {image_path}: {e}")
        return None


def compute_hashes_for_paths(
    paths: List[Path], max_workers: int = 8, desc: str = "Hashing images"
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Compute dHash and pHash concurrently for a collection of image paths."""
    valid_paths = []
    dhashes = []
    phashes = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(hash_single_image, paths), total=len(paths), desc=desc))

    for res in results:
        if res is not None:
            p_str, dh, ph = res
            valid_paths.append(p_str)
            dhashes.append(dh)
            phashes.append(ph)

    return (
        valid_paths,
        np.array(dhashes, dtype=np.uint8),
        np.array(phashes, dtype=np.uint8),
    )


def compute_pairwise_hamming_distance(
    hashes1: np.ndarray, hashes2: np.ndarray
) -> np.ndarray:
    """Vectorized pairwise Hamming distance between two sets of 64-bit boolean hashes.

    Args:
        hashes1: Array of shape (N, 64) with values in {0, 1}.
        hashes2: Array of shape (M, 64) with values in {0, 1}.

    Returns:
        Matrix of shape (N, M) containing integer Hamming distances in [0, 64].
    """
    return np.count_nonzero(hashes1[:, None, :] != hashes2[None, :, :], axis=2)


def audit_cross_dataset_visual_similarity(
    aptos_paths: List[Path],
    idrid_paths: List[Path],
    cache_path: Optional[Path] = None,
    max_workers: int = 8,
    near_duplicate_threshold: int = 5,
) -> Dict[str, Any]:
    """Audit visual similarity between APTOS and IDRiD cohorts."""
    # Check cache
    if cache_path and cache_path.exists():
        logger.info(f"Loading perceptual hashes from cache: {cache_path}")
        cache_data = np.load(cache_path, allow_pickle=True)
        aptos_names = list(cache_data["aptos_names"])
        aptos_dh = cache_data["aptos_dh"]
        aptos_ph = cache_data["aptos_ph"]
        idrid_names = list(cache_data["idrid_names"])
        idrid_dh = cache_data["idrid_dh"]
        idrid_ph = cache_data["idrid_ph"]
    else:
        logger.info(f"Computing perceptual hashes for APTOS ({len(aptos_paths)} images)...")
        aptos_names, aptos_dh, aptos_ph = compute_hashes_for_paths(
            aptos_paths, max_workers=max_workers, desc="Hashing APTOS"
        )

        logger.info(f"Computing perceptual hashes for IDRiD ({len(idrid_paths)} images)...")
        idrid_names, idrid_dh, idrid_ph = compute_hashes_for_paths(
            idrid_paths, max_workers=max_workers, desc="Hashing IDRiD"
        )

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                aptos_names=np.array(aptos_names),
                aptos_dh=aptos_dh,
                aptos_ph=aptos_ph,
                idrid_names=np.array(idrid_names),
                idrid_dh=idrid_dh,
                idrid_ph=idrid_ph,
            )
            logger.info(f"Saved perceptual hashes to cache: {cache_path}")

    logger.info("Computing cross-dataset Hamming distance matrices...")
    phash_dists = compute_pairwise_hamming_distance(aptos_ph, idrid_ph)
    dhash_dists = compute_pairwise_hamming_distance(aptos_dh, idrid_dh)

    min_phash = int(np.min(phash_dists))
    min_dhash = int(np.min(dhash_dists))

    # Candidate identification
    phash_candidates = []
    near_p_coords = np.argwhere(phash_dists <= near_duplicate_threshold)
    for a_idx, i_idx in near_p_coords:
        phash_candidates.append({
            "aptos_image": Path(aptos_names[a_idx]).name,
            "idrid_image": Path(idrid_names[i_idx]).name,
            "phash_distance": int(phash_dists[a_idx, i_idx]),
            "dhash_distance": int(dhash_dists[a_idx, i_idx]),
        })

    dhash_candidates = []
    near_d_coords = np.argwhere(dhash_dists <= 1)  # examine distance <= 1
    for a_idx, i_idx in near_d_coords:
        dhash_candidates.append({
            "aptos_image": Path(aptos_names[a_idx]).name,
            "idrid_image": Path(idrid_names[i_idx]).name,
            "phash_distance": int(phash_dists[a_idx, i_idx]),
            "dhash_distance": int(dhash_dists[a_idx, i_idx]),
        })

    # Exact duplicate definition: BOTH frequency and gradient distance must be 0
    exact_duplicates = np.sum((phash_dists == 0) & (dhash_dists == 0))

    # Joint near duplicate: both pHash <= 5 AND dHash <= 5
    joint_near_duplicates = int(np.sum((phash_dists <= near_duplicate_threshold) & (dhash_dists <= near_duplicate_threshold)))

    report = {
        "aptos_count": len(aptos_names),
        "idrid_count": len(idrid_names),
        "total_pairwise_comparisons": int(phash_dists.size),
        "phash_stats": {
            "min_distance": min_phash,
            "mean_distance": float(np.mean(phash_dists)),
            "std_distance": float(np.std(phash_dists)),
            "p1": float(np.percentile(phash_dists, 1)),
            "p5": float(np.percentile(phash_dists, 5)),
            "p25": float(np.percentile(phash_dists, 25)),
            "p50": float(np.percentile(phash_dists, 50)),
        },
        "dhash_stats": {
            "min_distance": min_dhash,
            "mean_distance": float(np.mean(dhash_dists)),
            "std_distance": float(np.std(dhash_dists)),
            "p1": float(np.percentile(dhash_dists, 1)),
            "p5": float(np.percentile(dhash_dists, 5)),
            "p25": float(np.percentile(dhash_dists, 25)),
            "p50": float(np.percentile(dhash_dists, 50)),
        },
        "exact_duplicates_count": int(exact_duplicates),
        "joint_near_duplicates_count": joint_near_duplicates,
        "phash_near_candidates_count": len(phash_candidates),
        "phash_candidates": phash_candidates[:50],
        "dhash_low_candidates_count": len(dhash_candidates),
        "dhash_low_candidates": dhash_candidates[:50],
        "is_visually_independent": bool(exact_duplicates == 0 and joint_near_duplicates == 0 and min_phash > near_duplicate_threshold),
    }

    return report
