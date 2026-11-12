"""A013_exponential_decay: Elementwise exponential decay."""
import numpy as np
import asnumpy as ap


def build_inputs():
    rng = np.random.default_rng(13)
    return {
        "x": rng.standard_normal((4_000_000,), dtype=np.float32),
    }


def forward(x):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    __numpilot_v_1 = ap.square(x)
    __numpilot_v_2 = ap.negative(__numpilot_v_1)
    __numpilot_v_3 = ap.exp(__numpilot_v_2)
    return {'out': __numpilot_v_3}