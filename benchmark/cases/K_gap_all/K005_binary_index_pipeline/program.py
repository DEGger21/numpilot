"""K005_binary_index_pipeline: Binary words, packed bit planes and sparse index remapping.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Integer results require exact comparison; NumPy >= 2.0 for bitwise_count.
The 17-column extent deliberately leaves partially filled packed bytes.
"""
import numpy as np


def build_inputs():
    rng = np.random.default_rng(1105)
    return {
        'words': rng.integers(0, 65536, (12, 17), dtype=np.uint32),
        'mask': np.array(21930, dtype=np.uint32),
    }


def transform_words(words, mask):
    """Bitwise operations preserve exact uint32 word semantics."""
    mixed = np.bitwise_xor(words, mask)
    low = np.bitwise_and(mixed, 255)
    high = np.right_shift(mixed, 8)
    shifted = np.bitwise_or(np.left_shift(low, 8), high)
    inverted = np.invert(mixed)
    counts = np.bitwise_count(mixed)
    return mixed, counts, shifted, inverted


def bit_planes(words):
    """Pack an odd-width bit plane and recover sparse coordinates."""
    plane = np.bitwise_and(words, 1).astype(bool)
    packed = np.packbits(plane, axis=1, bitorder="little")
    restored = np.unpackbits(packed, axis=1, count=17, bitorder="little")
    positions = np.flatnonzero(restored)
    coordinate_tuple = np.unravel_index(positions, restored.shape)
    coordinates = np.stack(coordinate_tuple, axis=1)
    return packed, restored, positions, coordinates


def count_patterns(counts):
    """Population-count grouping and count histogram."""
    flattened = np.ravel(counts)
    unique, inverse, frequencies = np.unique(flattened, return_inverse=True, return_counts=True)
    histogram = np.bincount(flattened, minlength=33)
    indices = np.searchsorted(unique, flattened)
    restored = np.take(unique, indices)
    return unique, inverse, frequencies, histogram, restored


def sparse_selection(words, counts):
    """Gather three high-population words from every row."""
    order = np.argpartition(counts, -3, axis=1)[:, -3:]
    selected = np.take_along_axis(words, order, axis=1)
    coordinates = np.argwhere(counts >= 10)
    rows = np.take(coordinates, 0, axis=1)
    columns = np.take(coordinates, 1, axis=1)
    return rows, columns, selected, order


def layout_views(words):
    """Shape manipulation retains the exact transformed bit patterns."""
    flipped = np.flip(words, axis=1)
    rotated = np.rot90(words, k=2)
    stacked = np.stack((words, flipped, rotated))
    repeated = np.repeat(words, 2, axis=0)
    tiled = np.tile(words, (1, 2))
    return stacked, repeated, tiled, flipped


def rank_counts(counts):
    """Distribution summaries include a padded transition signal."""
    quantiles = np.quantile(counts, [0.1, 0.5, 0.9], axis=1)
    partitioned = np.partition(counts, 8, axis=1)
    median = np.median(counts, axis=1)
    transitions = np.diff((counts > 8).astype(np.int32), axis=1)
    boundaries = np.pad(transitions, ((0, 0), (1, 0)))
    return quantiles, partitioned, median, boundaries


def forward(words, mask):
    mixed, counts, shifted, inverted = transform_words(words, mask)
    packed, restored, positions, coordinates = bit_planes(mixed)
    unique, inverse, frequencies, histogram, restored_counts = count_patterns(counts)
    rows, columns, selected, order = sparse_selection(mixed, counts)
    stacked, repeated, tiled, flipped = layout_views(shifted)
    quantiles, partitioned, median, boundaries = rank_counts(counts)
    intersection = np.intersect1d(positions, np.arange(0, words.size, 2))
    difference = np.setdiff1d(positions, intersection)
    return {
        'mixed': mixed,
        'counts': counts,
        'shifted': shifted,
        'inverted': inverted,
        'packed': packed,
        'restored': restored,
        'positions': positions,
        'coordinates': coordinates,
        'unique': unique,
        'inverse': inverse,
        'frequencies': frequencies,
        'histogram': histogram,
        'restored_counts': restored_counts,
        'rows': rows,
        'columns': columns,
        'selected': selected,
        'order': order,
        'stacked': stacked,
        'repeated': repeated,
        'tiled': tiled,
        'flipped': flipped,
        'quantiles': quantiles,
        'partitioned': partitioned,
        'median': median,
        'boundaries': boundaries,
        'intersection': intersection,
        'difference': difference,
    }
