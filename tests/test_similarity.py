"""Unit tests for perceptual hash and visual duplicate verification."""

from __future__ import annotations

import numpy as np
import pytest

from src.similarity import (
    compute_dhash,
    compute_phash,
    compute_pairwise_hamming_distance,
)


def test_phash_and_dhash_exact_match():
    """Identical images must yield 0 Hamming distance for both pHash and dHash."""
    img = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)

    ph1 = compute_phash(img)
    ph2 = compute_phash(img)
    dh1 = compute_dhash(img)
    dh2 = compute_dhash(img)

    assert len(ph1) == 64
    assert len(dh1) == 64
    assert np.array_equal(ph1, ph2)
    assert np.array_equal(dh1, dh2)


def test_phash_near_duplicate_robustness():
    """Slight brightness/noise perturbation should produce low Hamming distance (<= 5)."""
    # Create base image with some patterns
    base = np.zeros((128, 128, 3), dtype=np.uint8)
    base[30:90, 30:90] = 180
    base[45:75, 45:75] = 240

    # Perturbed version with slight brightness shift
    perturbed = np.clip(base.astype(np.int16) + 10, 0, 255).astype(np.uint8)

    ph_base = compute_phash(base)
    ph_pert = compute_phash(perturbed)

    dist = np.count_nonzero(ph_base != ph_pert)
    assert dist <= 5, f"pHash distance {dist} should be <= 5 for minor brightness perturbation."


def test_pairwise_hamming_distance():
    """Verify matrix calculation of pairwise Hamming distance."""
    # 3 images in set A, 2 images in set B
    h_a = np.array([
        [0] * 64,  # all 0s
        [1] * 64,  # all 1s
        [0, 1] * 32,  # 32 zeros, 32 ones
    ], dtype=np.uint8)

    h_b = np.array([
        [0] * 64,  # all 0s
        [1] * 64,  # all 1s
    ], dtype=np.uint8)

    dist_matrix = compute_pairwise_hamming_distance(h_a, h_b)

    assert dist_matrix.shape == (3, 2)
    # Row 0 vs [0s, 1s] -> [0, 64]
    assert dist_matrix[0, 0] == 0
    assert dist_matrix[0, 1] == 64
    # Row 1 vs [0s, 1s] -> [64, 0]
    assert dist_matrix[1, 0] == 64
    assert dist_matrix[1, 1] == 0
    # Row 2 vs [0s, 1s] -> [32, 32]
    assert dist_matrix[2, 0] == 32
    assert dist_matrix[2, 1] == 32

