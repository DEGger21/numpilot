import asnumpy as ap

x = ap.full((1024, 1024), 0.5, dtype=ap.float32)

# NUMPILOT_DEVICE_INPUTS_END

__numpilot_v_1 = ap.sum(x, axis=1)
__numpilot_v_2 = ap.full_like(__numpilot_v_1, 10.0, dtype=__numpilot_v_1.dtype)
__numpilot_v_3 = ap.add(__numpilot_v_1, __numpilot_v_2)
__numpilot_v_4 = ap.square(__numpilot_v_3)

out = __numpilot_v_3
squared = __numpilot_v_4
