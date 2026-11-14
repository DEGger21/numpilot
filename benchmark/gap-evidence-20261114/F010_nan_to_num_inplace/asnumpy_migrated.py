import asnumpy as ap

x = ap.full((1024, 1024), 0.5, dtype=ap.float32)

# NUMPILOT_DEVICE_INPUTS_END

__numpilot_v_1 = ap.nan_to_num(x, nan=3.0, posinf=4.0, neginf=-4.0)
__numpilot_v_2 = ap.square(__numpilot_v_1)

out = __numpilot_v_1
input_after = __numpilot_v_1
squared = __numpilot_v_2
