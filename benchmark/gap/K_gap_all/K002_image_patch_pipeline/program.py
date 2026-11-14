"""K002_image_patch_pipeline: Image patch statistics, resampling and boundary analysis.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
The image has an asymmetric shape and a planted bright rectangle.
All computation APIs are gaps; slicing and arithmetic remain ordinary Python/NumPy.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(1102)
    image = rng.uniform(0, 1, (18, 22)).astype(np.float32)
    image[5:10, 7:14] += 1.0
    return {
        'image': image,
        'thresholds': np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float32),
    }


def patch_statistics(image):
    """Rank statistics over reflected three-by-three neighborhoods."""
    halo = np.pad(image, ((1, 1), (1, 1)), mode="reflect")
    patches = np.lib.stride_tricks.sliding_window_view(halo, (3, 3))
    median = np.median(patches, axis=(-2, -1))
    upper = np.percentile(patches, 75, axis=(-2, -1))
    lower = np.percentile(patches, 25, axis=(-2, -1))
    return median, upper, lower, patches


def gradients(image):
    """First differences and a periodic curvature proxy."""
    dy, dx = np.gradient(image)
    edges = (dx * dx + dy * dy) ** 0.5
    horizontal = np.diff(dx, axis=1)
    vertical = np.diff(dy, axis=0)
    padded_h = np.pad(horizontal, ((0, 0), (0, 1)))
    padded_v = np.pad(vertical, ((0, 1), (0, 0)))
    curvature = padded_h + padded_v
    return edges, dx, dy, curvature


def augment(image):
    """Shape-preserving flips and a two-by-two mosaic."""
    horizontal = np.flip(image, axis=1)
    vertical = np.flip(image, axis=0)
    rotated = np.rot90(image, k=2)
    views = np.stack((image, horizontal, vertical, rotated), axis=0)
    mosaic = np.block([[image, horizontal], [vertical, rotated]])
    mirrored = np.concatenate((image, horizontal), axis=1)
    return views, mosaic, mirrored


def resize(image):
    """Nearest-neighbor expansion and explicit indexed decimation."""
    expanded_y = np.repeat(image, 2, axis=0)
    expanded = np.repeat(expanded_y, 3, axis=1)
    rows = np.arange(0, image.shape[0], 2)
    columns = np.arange(0, image.shape[1], 2)
    selected_rows = np.take(image, rows, axis=0)
    decimated = np.take(selected_rows, columns, axis=1)
    tiled = np.tile(decimated, (2, 2))
    return expanded, decimated, tiled


def classify(image, thresholds):
    """Threshold classes and non-byte-aligned boolean packing."""
    labels = np.digitize(image, thresholds)
    counts = np.bincount(labels.ravel(), minlength=5)
    packed = np.packbits(labels > 1, axis=1, bitorder="little")
    restored = np.unpackbits(packed, axis=1, count=image.shape[1], bitorder="little")
    return labels, counts, packed, restored


def select_edges(edges):
    """Quantile threshold with coordinates and gathered strengths."""
    threshold = np.quantile(edges, 0.9)
    coordinates = np.argwhere(edges >= threshold)
    positions = np.ravel_multi_index(coordinates.T, edges.shape)
    strengths = np.take(edges, positions)
    quantiles = np.quantile(strengths, [0.25, 0.5, 0.75])
    return coordinates, strengths, quantiles


def forward(image, thresholds):
    median, upper, lower, patches = patch_statistics(image)
    edges, dx, dy, curvature = gradients(median)
    views, mosaic, mirrored = augment(median)
    resized, decimated, tiled = resize(median)
    labels, counts, packed, restored = classify(median, thresholds)
    coordinates, strengths, quantiles = select_edges(edges)
    column_bands = np.split(median, [6, 12], axis=1)
    joined = np.concatenate(column_bands, axis=1)
    ranked_rows = np.partition(median, median.shape[1] // 2, axis=1)
    return {
        'median': median,
        'upper': upper,
        'lower': lower,
        'edges': edges,
        'dx': dx,
        'dy': dy,
        'curvature': curvature,
        'views': views,
        'mosaic': mosaic,
        'mirrored': mirrored,
        'resized': resized,
        'decimated': decimated,
        'tiled': tiled,
        'labels': labels,
        'counts': counts,
        'packed': packed,
        'restored': restored,
        'coordinates': coordinates,
        'strengths': strengths,
        'quantiles': quantiles,
        'joined': joined,
        'ranked_rows': ranked_rows,
    }
