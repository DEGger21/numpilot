"""G004_cbrt_three_channels: Signed compression of three channels.

Inputs are constructed on the CPU; forward returns named NumPy arrays.
Three static call sites share exactly one contract:
cbrt|float32|(4,8)
"""
import numpy as np
import asnumpy as ap

def build_inputs():
    rng = np.random.default_rng(704)
    return {'a': rng.standard_normal((4, 8), dtype=np.float32), 'b': rng.standard_normal((4, 8), dtype=np.float32), 'c': rng.standard_normal((4, 8), dtype=np.float32)}

def forward(a, b, c):
    if isinstance(a, np.ndarray):
        a = ap.ndarray.from_numpy(a)
    if isinstance(b, np.ndarray):
        b = ap.ndarray.from_numpy(b)
    if isinstance(c, np.ndarray):
        c = ap.ndarray.from_numpy(c)
    __numpilot_v_1 = ap.copysign(ap.power(ap.absolute(a), ap.full_like(a, 0.3333333333333333, dtype=a.dtype)), a)
    __numpilot_v_2 = ap.copysign(ap.power(ap.absolute(b), ap.full_like(b, 0.3333333333333333, dtype=b.dtype)), b)
    __numpilot_v_3 = ap.copysign(ap.power(ap.absolute(c), ap.full_like(c, 0.3333333333333333, dtype=c.dtype)), c)
    __numpilot_v_4 = ap.multiply(__numpilot_v_1, ap.full(__numpilot_v_1.shape, 0.5, dtype=__numpilot_v_1.dtype))
    __numpilot_v_5 = ap.multiply(__numpilot_v_2, ap.full(__numpilot_v_2.shape, 0.3, dtype=__numpilot_v_2.dtype))
    __numpilot_v_6 = ap.add(__numpilot_v_4, __numpilot_v_5)
    __numpilot_v_7 = ap.multiply(__numpilot_v_3, ap.full(__numpilot_v_3.shape, 0.2, dtype=__numpilot_v_3.dtype))
    __numpilot_v_8 = ap.add(__numpilot_v_6, __numpilot_v_7)
    __numpilot_v_9 = ap.tanh(__numpilot_v_8)
    __numpilot_v_10 = ap.multiply(__numpilot_v_1, __numpilot_v_2)
    __numpilot_v_11 = ap.multiply(__numpilot_v_10, __numpilot_v_3)
    return {'out': __numpilot_v_9, 'agreement': __numpilot_v_11, 'compressed_a': __numpilot_v_1}