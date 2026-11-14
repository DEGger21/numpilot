import asnumpy as ap

x = ap.full((1024, 1024), 0.5, dtype=ap.float32)
y = ap.full((1024, 1024), 0.5, dtype=ap.float32)
buffer = ap.full((1024, 1024), 0.5, dtype=ap.float32)

# NUMPILOT_DEVICE_INPUTS_END

__numpilot_v_1 = ap.add(x, y)
__numpilot_v_2 = ap.square(__numpilot_v_1)

out = __numpilot_v_1
norm = __numpilot_v_2
