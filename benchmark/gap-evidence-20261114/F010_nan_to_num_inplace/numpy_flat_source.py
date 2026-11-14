import numpy as np

x = np.full((1024, 1024), 0.5, dtype=np.float32)

# NUMPILOT_DEVICE_INPUTS_END

__numpilot_v_1 = np.nan_to_num(x, nan=3.0, posinf=4.0, neginf=-4.0)
__numpilot_v_2 = np.square(__numpilot_v_1)

out = __numpilot_v_1
input_after = __numpilot_v_1
squared = __numpilot_v_2
