import numpy as np


def frame_indices(length, phase=0.5):
    """Uniform endpoint-inclusive clip. Offset within each interior frame's bin."""
    if length < 1 or not 0 <= phase < 1:
        raise ValueError((length, phase))
    canonical = np.rint(np.linspace(0, length - 1, 64)).astype(np.int64)
    if length <= 64 or phase == 0.5:
        return canonical
    centers = canonical[1:-1]
    edges = np.r_[1, (centers[:-1] + centers[1:]) // 2 + 1, length - 1]
    if phase < 0.5:
        interior = edges[:-1] + np.floor(2 * phase * (centers - edges[:-1])).astype(
            np.int64
        )
    else:
        interior = centers + np.floor((2 * phase - 1) * (edges[1:] - centers)).astype(
            np.int64
        )
    return np.r_[0, interior, length - 1]
