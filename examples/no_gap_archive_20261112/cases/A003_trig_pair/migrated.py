"""A003_trig_pair: Sine and cosine of independent angles."""
import numpy as np
import asnumpy as ap

def build_inputs():
    rng = np.random.default_rng(3)
    return {'x': rng.standard_normal((48,), dtype=np.float32)}

def forward(x):
    if isinstance(x, np.ndarray):
        x = ap.ndarray.from_numpy(x)
    __numpilot_v_1 = ap.sin(x)
    __numpilot_v_2 = ap.cos(x)
    return {'sin': __numpilot_v_1, 'cos': __numpilot_v_2}