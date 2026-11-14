import asnumpy as ap

offset = ap.full((1048576,), 0.5, dtype=ap.float32)

# NUMPILOT_DEVICE_INPUTS_END

__numpilot_v_1 = ap.linspace(0.0, 0.9999990463256836, 1048576, dtype=ap.float32)
__numpilot_v_2 = ap.add(__numpilot_v_1, offset)

out = __numpilot_v_2
