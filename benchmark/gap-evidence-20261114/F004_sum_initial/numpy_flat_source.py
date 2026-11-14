import numpy as np

x = np.full((1024, 1024), 0.5, dtype=np.float32)

# NUMPILOT_DEVICE_INPUTS_END

__numpilot_v_1 = np.sum(x, axis=1)
__numpilot_v_2 = np.full_like(__numpilot_v_1, 10.0, dtype=__numpilot_v_1.dtype)
__numpilot_v_3 = np.add(__numpilot_v_1, __numpilot_v_2)
__numpilot_v_4 = np.square(__numpilot_v_3)

out = __numpilot_v_3
squared = __numpilot_v_4
